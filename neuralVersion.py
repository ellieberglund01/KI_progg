import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class CONSYS:
    def __init__(self, controller, plant,learning_rate):
        self.controller = controller
        self.plant = plant
        self.learning_rate = learning_rate

    def run_system(self, num_epochs, timesteps):
        if isinstance(self.controller, NeuralController):
            self.controller.layers = [3] + self.controller.layers + [1]
            mse_history = self.jaxrun(num_epochs)

        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) 
        mse_history = []
        for epoch in range (1,num_epochs):
            self.plant.reset()
            self.controller.error_history = jnp.zeros(timesteps)
            mse, gradients = gradfunc(self.controller.get_params(), timesteps) 
            mse_history.append(mse)
            self.controller.update_params(self.learning_rate, gradients)
        
        self.controller.plot(mse_history)
        return mse_history

    def run_one_epoch(self, params, timesteps):
        U = 0
        if isinstance(self.controller, NeuralController):
            all_features = jnp.array([])  
            all_targets = jnp.array([])  

        prev_error = 0
        integral = 0

        for t in range(timesteps):
            plant_output = self.plant.update(U) 
            error = self.plant.target - plant_output  
            derivative = error - prev_error
            integral += error
            prev_error = error
            if isinstance(self.controller, NeuralController):
                features = jnp.array([error, derivative, integral])
                jnp.append(all_features,features)
                jnp.append(all_targets, error)
                U = self.controller.predict(params,features).reshape(-1)[0]
            else: 
                U = self.controller.compute(params, error, t) 

        if isinstance(self.controller, NeuralController):
            return all_features, all_targets

        else:
            mse = jnp.mean(jnp.square(self.controller.error_history))
            return mse

    
    def jaxrun(self, num_epochs):
        params = self.controller.get_jaxnet_params(self.controller.layers)
        mse_history = self.jaxnet_train(params, num_epochs,learning_rate)
        return mse_history

    def jaxnet_train(self,params, num_epochs,learning_rate):
        mse_history = []
        curr_params = params
        for _ in range(num_epochs):
            curr_params, mse = self.jaxnet_train_one_epoch(curr_params,learning_rate)
            mse_history.append(mse)
            self.controller.params = curr_params
        return mse_history 
    
    def jaxnet_train_one_epoch(self,params,learning_rate):
        mse, gradients = jax.value_and_grad(self.controller.jaxnet_loss)(params, self)
        return [(w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, gradients)], mse
    

class PIDController:
    def __init__(self, kp, ki, kd, timesteps):
        self.params = jnp.array([kp,ki,kd])
        self.error_history = jnp.zeros(timesteps)
        self.k1_his = []  # Store changes in k1 across epochs
        self.k2_his = []  # Store changes in k2 across epochs
        self.k3_his = []  # Store changes in k3 across epochs
        
    def get_params(self):
        return self.params
    
    def compute(self,params, error, timestep):
        integral = jnp.sum(self.error_history)
        derivative = error-self.error_history[timestep-1]
        kp = params[0]
        ki = params[1]
        kd = params[2]
        self.error_history = self.error_history.at[timestep].set(error)
        p = kp*error
        d = kd*derivative
        i = ki*integral
        U = p+d+i
        return U
    
    def update_params(self,learning_rate, gradients):
        self.params = [param-learning_rate*grad for param, grad in zip(self.params,gradients)]

        self.k1_his.append(self.params[0])
        self.k2_his.append(self.params[1])
        self.k3_his.append(self.params[2])

    def plot(self, mse_history):
        # Plot the progression of learning
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs), mse_history, marker='o', linestyle='-')
        plt.title('Progression of Learning')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.show()

         # Plot the changes in k parameters across epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs), self.k1_his, marker='o', label='k1')
        plt.plot(range(1, num_epochs), self.k2_his, marker='o', label='k2')
        plt.plot(range(1, num_epochs), self.k3_his, marker='o', label='k3')
        plt.title('Changes to k parameters across epochs')
        plt.xlabel('Epochs')
        plt.ylabel('k-value')
        plt.legend()
        plt.grid(True)
        plt.show()

class NeuralController:
    def __init__(self, layers, timesteps, lower, upper):
        self.params = []
        self.layers = layers
        self.timesteps = timesteps
        self.error_history = jnp.zeros(timesteps)
        self.lower = lower
        self.upper = upper
        #self.activiation_functions = activation_functions
        
    def get_jaxnet_params(self, layers): 
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(self.lower,self.upper,(sender,receiver))
            biases = np.random.uniform(self.lower,self.upper,(1,receiver))
            sender = receiver
            params.append([weights, biases])
        self.params = params
        return params  
    
    def get_params(self):
        return self.params
   
    def jaxnet_loss(self, params,consys):
        features, targets = consys.run_one_epoch(params, self.timesteps) 
        batched_predict = jax.vmap(self.predict, in_axes=(None, 0)) #Se mer på
        predictions = batched_predict(params, features)
        return jnp.mean(jnp.square(targets - predictions)) 


    def predict(self, all_params, features): #dette er kun med sigmoid
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))
        activations = features
        for weights, biases in all_params:
            activations= sigmoid(jnp.dot(activations,weights) + biases)
        return activations
    

#PLANTS:
class BathTubPlant:
    def __init__(self, A, C, H_0, lower, upper):
        self.A = A
        self.C = C
        self.target = H_0
        self.H= H_0
        self.g = 9.8
        self.noise_range = [-0.1,0.1] 
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 


    def reset(self):
        self.__init__(self.A,self.C,self.target, self.rangeNoiseLower, self.rangeNoiseUpper)

    def update(self, U):
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper)
        V = jnp.sqrt(2 * self.g * self.H) 
        Q= V * self.C 
        change_in_volume = U + D - Q
        change_in_height = change_in_volume / self.A
        self.H += change_in_height
        return self.H
    
    def get_H(self):
        return self.H

class CournotPlant:
    def __init__(self, pmax, c_m, T, lower, upper, ):
        self.target = T
        self.pmax = pmax
        self.c_m = c_m
        self.q1 = 0.5
        self.q2 = 0.5
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 
        
    def reset(self):
        self.__init__(self.pmax,self.c_m,self.target,self.rangeNoiseLower, self.rangeNoiseUpper)

    def update(self,U): #Heller sette Disturbance i controlSys og ha som input her?
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper) 
        # Update quantities

        self.q1 += U
        self.q2 += D

        if self.q1 < 0:
            self.q1=0
        if self.q2 < 0:
            self.q2=0
        if self.q1 > 1:
            self.q1=1
        if self.q2 > 1:
            self.q2=1

        q = self.q1 + self.q2
        price = self.pmax - q
        P1 = self.q1 * (price - self.c_m)
        return P1   


#Funker ikke
class HeaterRoomPlant: #insulingregulering
    def __init__(self, initial_temp, target, lower_noise, upper_noise):
        self.temperature = initial_temp  # Initial room temperature

        self.lower_noise = lower_noise
        self.upper_noise = upper_noise

    def reset(self):
        self.__init__(self.temperature, self.lower_noise, self.upper_noise)

    def update(self, U):
        D = np.random.uniform(low=self.lower_noise, high=self.upper_noise)
        # Simple room temperature model
        delta_temperature = U / 10.0 + D
        self.temperature += delta_temperature
        return self.temperature

    def get_temperature(self):
        return self.temperature

#FLYTT DETTE TIL MAIN
kp = 1
ki = 0.1
kd = 0.01

num_epochs = 40
learning_rate = 0.001
timesteps = 40
rangeNoiseLower = -0.01
rangeNoiseUpper = 0.01
lower = -0.1
upper = 0.1

#Bathtub
A=100
C=1
H0=1

#Cournotplant
pmax = 10
c_m = 0.1 
T = 30

#Heaterroom
initial_temp = 20

bathtub = BathTubPlant(A=A,C=C,H_0= H0, lower=rangeNoiseLower, upper=rangeNoiseUpper)
cournotPlant = CournotPlant( pmax=pmax, c_m =c_m, T=T, lower=rangeNoiseLower, upper=rangeNoiseUpper)
#heater_room_plant = HeaterRoomPlant(initial_temp, rangeNoiseLower, rangeNoiseUpper)
 
pid_controller = PIDController(kp=kp,ki=ki,kd=ki, timesteps=timesteps)
neural_controller = NeuralController(layers = [5,10,5], timesteps=timesteps, lower=lower, upper=upper)

consys = CONSYS(controller=pid_controller,plant=cournotPlant,learning_rate=learning_rate)
consys.run_system(num_epochs,timesteps)

