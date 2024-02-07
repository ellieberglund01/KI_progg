import numpy as np
import copy
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
            self.controller.get_jaxnet_params(self.controller.layers)
        
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) 
        gradfunc = jax.jit(gradfunc, static_argnums=[1])
        mse_history = []

        for epoch in range(1,num_epochs):
            self.plant.reset() #Gjør nada
            self.controller.error_history = jnp.zeros(timesteps)
            mse, gradients = gradfunc(self.controller.get_params(), timesteps) #get_params?
            mse_history.append(mse)
            print(epoch, mse)
            self.controller.update_params(self.learning_rate,gradients)
            #params = self.controller.params
                
        self.controller.plot(mse_history)
        self.controller.plotk()
        return mse_history

    def run_one_epoch(self, params, timesteps):
        U = 0
        prev_error = 0
        integral = 0
        features = jnp.array([])
        plant = copy.deepcopy(self.plant)
        sse=0

        for t in range(timesteps):
            plant_output = plant.update(U) 
            error = plant.target - plant_output  
            derivative = error - prev_error
            integral += error
            prev_error = error
            sse += error**2
           
            features = jnp.array([error, derivative, integral]) 
            U = self.controller.compute(params, features, t).reshape(-1)[0]
        mse=sse/timesteps        
        #mse = jnp.mean(jnp.square(self.controller.error_history))
        return mse

class Controller:
    def __init__(self, timesteps):
        self.timesteps = timesteps
    def run_one_epoch(self, params, timesteps):
        raise NotImplementedError("Must be implemented by subclass.")
    def update_params(self,learning_rate, gradient):
        raise NotImplementedError("Must be implemented by subclass.")

    def plot(self, mse_history):
    # Plot the progression of learning
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs), mse_history, marker='o', linestyle='-')
        plt.title('Progression of Learning')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.show()

    def plotk(self):
        pass


class PIDController(Controller):
    def __init__(self, kp, ki, kd, timesteps):
        super().__init__(timesteps)
        self.params = jnp.array([kp,ki,kd])
        self.error_history = jnp.zeros(timesteps)
        self.k1_his = []  # Store changes in k1 across epochs
        self.k2_his = []  # Store changes in k2 across epochs
        self.k3_his = []  # Store changes in k3 across epochs
        
    def get_params(self):
        return self.params
    
    def compute(self,params, features, timestep):
        kp = params[0]
        ki = params[1]
        kd = params[2]
        #self.error_history = self.error_history.at[timestep].set(features[0])
        p = kp*features[0]
        d = kd*features[1]
        i = ki*features[2]
        U = p+d+i
        return U
    
    def plotk(self):
        # Plot the changes in k parameters across epochs (Må bare være for PID)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs), self.k1_his, marker='o', label='kp') #hvordan
        plt.plot(range(1, num_epochs), self.k2_his, marker='o', label='ki')
        plt.plot(range(1, num_epochs), self.k3_his, marker='o', label='kd')
        plt.title('Changes to k parameters across epochs')
        plt.xlabel('Epochs')
        plt.ylabel('k-value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def update_params(self,learning_rate, gradients):
        self.params = [param-learning_rate*grad for param, grad in zip(self.params,gradients)]

        self.k1_his.append(self.params[0])
        self.k2_his.append(self.params[1])
        self.k3_his.append(self.params[2])


class NeuralController(Controller):
    def __init__(self, layers, timesteps, activation_function, lower, upper):
        super().__init__(timesteps)
        self.params = jnp.array([])
        self.layers = layers
        self.timesteps = timesteps
        self.activation_function = self.get_activation_function(activation_function)
        self.error_history = jnp.zeros(timesteps)
        self.lower = lower
        self.upper = upper
        #self.activiation_functions = activation_functions

    def get_activation_function(self,name):
        if name.lower() == "sigmoid":
            return lambda x: 1 / (1 + jnp.exp(-x))
        elif name.lower() == "relu":
            return lambda x: jnp.maximum(0, x)
        elif name.lower() == "tanh":
            return lambda x: jnp.tanh(x)
        else:
            raise ValueError(f"Activation function '{name}' is n ot supported.")
        
    def get_jaxnet_params(self, layers): #jax format?
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(self.lower,self.upper,(sender,receiver))
            biases = np.random.uniform(self.lower,self.upper,(1,receiver))
            params.append([weights,biases])
            sender = receiver
        self.params = [(jnp.array(w), jnp.array(b)) for w, b in params]
        return params  
    
    def get_params(self):
        return self.params
    
    def update_params(self, learning_rate, gradients):
        self.params = [(w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(self.params, gradients)]
   
    def compute(self, all_params, features, timestep): 
        activations = features 
        for i, (weights, biases) in enumerate(all_params):
            if i == len(all_params) - 1:
                # Remove sigmoid activation from the last layer (output layer)
                activations = jnp.dot(activations, weights) + biases
            else:
                z = jnp.dot(activations, weights) + biases
                activations = self.activation_function(z)
        #self.error_history = self.error_history.at[timestep].set(features[0])
        return activations

#PLANTS:
class BathTubPlant:
    def __init__(self, A, C, H_0, lower, upper):
        self.A = A
        self.C = C
        self.target = H_0
        self.H = H_0
        self.g = 9.8
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 

    def reset(self):
        self.__init__(self.A,self.C,self.target, self.rangeNoiseLower, self.rangeNoiseUpper)

    def update(self, U):
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper)
        V = jnp.sqrt(2 * self.g * self.H) 
        Q = V * self.C 
        change_in_volume = U + D - Q
        change_in_height = change_in_volume / self.A
        self.H += change_in_height
        return self.H
    
    def get_H(self):
        return self.H

class CournotPlant:
    def __init__(self, pmax, c_m, T, lower, upper):
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

        self.q1 = jnp.clip(self.q1,0,1)
        self.q2 = jnp.clip(self.q2,0,1)

        q = self.q1 + self.q2
        price = self.pmax - q
        P1 = self.q1 * (price - self.c_m)
        return P1   


#Funker ikke
class HeaterRoomPlant: 
    def __init__(self, initial_temp, target, lower_noise, upper_noise):
        self.temperature = initial_temp  # Initial room temperature
        self.lower_noise = lower_noise
        self.upper_noise = upper_noise
        self.target = target

    def reset(self):
        self.__init__(self.temperature, self.target, self.lower_noise, self.upper_noise)

    def update(self, U):
        D = np.random.uniform(low=self.lower_noise, high=self.upper_noise)
        delta_temperature = jnp.array(U) / 5.0 + D
        new_temperature = self.temperature + delta_temperature
        # Create a new instance of the object with the updated temperature
        updated_plant = HeaterRoomPlant(new_temperature, self.target, self.lower_noise, self.upper_noise)
        return updated_plant.temperature

    def get_temperature(self):
        return self.temperature


#FLYTT DETTE TIL MAIN
kp = 0.001
ki = 0.001
kd = 0.001

num_epochs = 1000
timesteps = 10
learning_rate = 0.0001
rangeNoiseLower = -0.01
rangeNoiseUpper = 0.01
lower = -0.01
upper = 0.01

#Bathtub
A=100
C=1
H0=1

#Cournotplant
pmax = 3.5
c_m = 0.1
T = 3.2

#Heaterroom
initial_temp = 20
target = 22

bathtub = BathTubPlant(A=A,C=C,H_0= H0, lower=rangeNoiseLower, upper=rangeNoiseUpper)
cournotPlant = CournotPlant( pmax=pmax, c_m =c_m, T=T, lower=rangeNoiseLower, upper=rangeNoiseUpper)
heater_room_plant = HeaterRoomPlant(initial_temp, target, rangeNoiseLower, rangeNoiseUpper)
 
pid_controller = PIDController(kp=kp,ki=ki,kd=ki, timesteps=timesteps)
neural_controller = NeuralController(layers = [5,10,5,3], timesteps=timesteps, activation_function="sigmoid",lower=lower, upper=upper)
consys = CONSYS(controller=neural_controller,plant=cournotPlant,learning_rate=learning_rate)
consys.run_system(num_epochs,timesteps)

