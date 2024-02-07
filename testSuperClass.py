
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


class CONSYS:
    def __init__(self, controller,plant, learning_rate):
        self.controller = controller
        self.plant = plant
        self.learning_rate = learning_rate

    def run_system(self, num_epochs, timesteps):
        if isinstance(self.controller, NeuralController):
            self.controller.layers = [3] + self.controller.layers + [1]
            mse_history = self.jaxrun(num_epochs)

        gradfunc = jax.value_and_grad(self.controller.run_one_epoch, argnums=0) 
        mse_history = []
        
        for epoch in range (1,num_epochs):
            self.plant.reset()
            self.controller.error_history = jnp.zeros(timesteps)
            mse, gradients = gradfunc(self.controller.get_params(), timesteps) 
            mse_history.append(mse)
            self.controller.update_params(self.learning_rate, gradients)    
        
        self.controller.plot(mse_history)
        if isinstance(self.controller, PIDController):
            self.controller.plotk

        return mse_history
    
    def jaxrun(self, num_epochs):
        new_params = self.controller.get_jaxnet_params(self.controller.layers)
        self.controller.params = new_params
        mse_history = self.jaxnet_train(new_params, num_epochs,learning_rate)
        return mse_history

    def jaxnet_train(self,params, num_epochs,learning_rate):
        mse_history = []
        curr_params = params
        for _ in range(num_epochs):
            curr_params, mse = self.jaxnet_train_one_epoch(params,learning_rate)
            self.controller.params = curr_params
            mse_history.append(mse)
        return mse_history 
    
    def jaxnet_train_one_epoch(self,params,learning_rate):
        mse, gradients = jax.value_and_grad(self.controller.jaxnet_loss)(params) 
        return [(w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, gradients)], mse
    
class Controller:
    def __init__(self, plant, timesteps):
        self.plant = plant
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
        # Plot the changes in k parameters across epochs (Må bare være for PID)
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

class PIDController(Controller):
    def __init__(self, kp, ki, kd, plant,timesteps):
        super().__init__(plant, timesteps)
        self.params = jnp.array([kp, ki, kd])
        self.error_history = jnp.zeros(timesteps)
        self.k1_his = []  
        self.k2_his = []  
        self.k3_his = [] 

    def run_one_epoch(self, params, timesteps):
        U = 0
        for t in range(timesteps):
            plant_output = self.plant.update(U) 
            error = self.plant.target - plant_output  
            U = self.compute(params, error, t)
        mse = jnp.mean(jnp.square(self.error_history))
        return mse 
        
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

    def update_params(self, learning_rate, gradients):
        self.params = [param-learning_rate*grad for param, grad in zip(self.params,gradients)]
        self.k1_his.append(self.params[0])
        self.k2_his.append(self.params[1])
        self.k3_his.append(self.params[2])


class NeuralController(Controller):
    def __init__(self, layers, activation_function, plant,timesteps, lower, upper):
        super().__init__(plant, timesteps)
        self.layers = layers
        self.lower, self.upper = lower, upper
        self.activation_function = self.get_activation_function(activation_function)
        self.params = self.get_jaxnet_params(layers)
    
    def get_activation_function(self,name):
        if name.lower() == "sigmoid":
            return lambda x: 1 / (1 + jnp.exp(-x))
        elif name.lower() == "relu":
            return lambda x: jnp.maximum(0, x)
        elif name.lower() == "tanh":
            return lambda x: jnp.tanh(x)
        else:
            raise ValueError(f"Activation function '{name}' is not supported.")
        
    def run_one_epoch(self, params, timesteps):
        U = 0
        feature_size = 3
        all_features_list = jnp.zeros((timesteps, feature_size))  
        all_targets_list = jnp.zeros(timesteps)
        prev_error = 0
        integral = 0

        for t in range(timesteps):
            plant_output = self.plant.update(U) 
            error = self.plant.target - plant_output  
            derivative = error - prev_error
            integral += error
            prev_error = error

            features = jnp.array([error, derivative, integral])
            all_features_list = all_features_list.at[t].set(features)
            all_targets_list = all_targets_list.at[t].set(error)
            U = self.predict(params,features).reshape(-1)[0]
        
        return all_features_list, all_targets_list


    def get_jaxnet_params(self, layers): 
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = jnp.array(np.random.uniform(self.lower,self.upper,(sender,receiver)))
            biases = jnp.array(np.random.uniform(self.lower,self.upper,(1,receiver)))
            sender = receiver
            params.append((weights, biases))
        return params  
    
    def get_params(self):
        return self.params  
    
    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    def predict(self, all_params, features):
        activations = features
        for weights, biases in all_params[:-1]:
            z = jnp.dot(activations, weights) + biases
            activations = self.activation_function(z)
        final_weights, final_biases = all_params[-1]
        activations = jnp.dot(activations, final_weights) + final_biases
        return activations

    batched_predict = jax.vmap(predict, in_axes=(None,None, 0))

    def jaxnet_loss(self, params):
        features, targets =self.run_one_epoch(params, self.timesteps) 
        predictions = self.batched_predict(params, features)
        return jnp.mean(jnp.square(targets - predictions)) 
    

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

    def update(self,U): 
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper) 
        self.q1 += U
        self.q2 += D

        self.q1 = max(0, min(self.q1, 1))
        self.q2 = max(0, min(self.q2, 1))

        q = self.q1 + self.q2
        price = self.pmax - q
        P1 = self.q1 * (price - self.c_m)
        return P1   


#FLYTT DETTE TIL MAIN
kp = 1
ki = 0.1
kd = 0.01

num_epochs = 100
learning_rate = 2
timesteps = 10
rangeNoiseLower = -0.01
rangeNoiseUpper = 0.01
lower = -0.1
upper = 0.1


#Bathtub
A=100
C=1
H0=1


#Cournotplant
pmax = 3.5
c_m = 0.01
T = 3.2


bathtub = BathTubPlant(A=A,C=C,H_0= H0, lower=rangeNoiseLower, upper=rangeNoiseUpper)
cournotPlant = CournotPlant( pmax=pmax, c_m =c_m, T=T, lower=rangeNoiseLower, upper=rangeNoiseUpper)


pid_controller = PIDController(kp=kp,ki=ki,kd=ki,plant=cournotPlant, timesteps=timesteps)
neural_controller = NeuralController(layers = [5,10,5],activation_function="relu", plant=bathtub, timesteps=timesteps, lower=lower, upper=upper)


consys = CONSYS(controller=neural_controller,plant=bathtub, learning_rate=learning_rate)
consys.run_system(num_epochs,timesteps)