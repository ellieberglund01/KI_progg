import numpy as np
import copy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json

class CONSYS:
    def __init__(self, controller, plant,learning_rate):
        self.controller = controller
        self.plant = plant
        self.learning_rate = learning_rate

    def run_system(self, num_epochs, timesteps):
        if isinstance(self.controller, NeuralController):
            self.controller.layers = [3] + self.controller.layers + [1] #setting up layers
            self.controller.gen_jaxnet_params(self.controller.layers) #generate parameters
        
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) #define tracing function
        gradfunc = jax.jit(gradfunc, static_argnums=[1]) #increase speed
        mse_history = []

        for epoch in range(1,num_epochs):
            mse, gradients = gradfunc(self.controller.get_params(), timesteps) #tracing starts. Gradients wrt params
            mse_history.append(mse)
            print(epoch, mse)
            self.controller.update_params(self.learning_rate,gradients) #updates params based on gradients and learning rate
                
        self.controller.plot(mse_history, num_epochs)
        self.controller.plotk(num_epochs)

    def run_one_epoch(self, params, timesteps):
        U = 0
        prev_error = 0
        integral = 0
        features = jnp.array([])
        plant = copy.deepcopy(self.plant) #Making deepcopy here so that when we modify plant, we don't modify self.plant
        sse=0 #squared standard error

        for t in range(timesteps): #calculates error each timestep
            plant_output = plant.update(U) 
            error = plant.target - plant_output  
            derivative = error - prev_error
            integral += error
            prev_error = error
            sse += error**2 #sum squared error fro each timestep
           
            features = jnp.array([error, derivative, integral]) 
            U = self.controller.compute(params, features).reshape(-1)[0] #Generate new U
        mse=sse/timesteps #mse for the epoch        
        return mse

class Controller:
    
    def get_params(self):
        return self.params
    
    def plot(self, mse_history, num_epochs):
    # Plot the progression of learning
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs), mse_history, linestyle='-')
        plt.title('Progression of Learning')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.show()

    def plotk(self, num_epochs):
        pass

class PIDController(Controller):
    def __init__(self, kp, ki, kd):
        super().__init__()
        self.params = jnp.array([kp,ki,kd])
        self.k1_his = []  # Store changes in k1 across epochs
        self.k2_his = []  # Store changes in k2 across epochs
        self.k3_his = []  # Store changes in k3 across epochs
    
    def compute(self,params, features):
        kp = params[0] 
        ki = params[1]
        kd = params[2]
 
        p = kp*features[0] #error
        d = kd*features[1] #derivative
        i = ki*features[2] #integral
        U = p+d+i
        return U
    
    def plotk(self, num_epochs): # Plot the changes in k parameters across epochs (Må bare være for PID)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs), self.k1_his, label='kp') 
        plt.plot(range(1, num_epochs), self.k2_his, label='ki')
        plt.plot(range(1, num_epochs), self.k3_his, label='kd')
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
    def __init__(self, layers, activation_function, lower, upper):
        super().__init__()
        self.params = jnp.array([])
        self.layers = layers
        self.activation_function = self.get_activation_function(activation_function)
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
        
    def gen_jaxnet_params(self, layers): #initializing weights + biases
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(self.lower,self.upper,(sender,receiver)) 
            biases = np.random.uniform(self.lower,self.upper,(1,receiver))
            params.append([weights,biases])
            sender = receiver
        self.params = [(jnp.array(w), jnp.array(b)) for w, b in params]
    
    def update_params(self, learning_rate, gradients): 
        self.params = [(w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(self.params, gradients)]
   
    def compute(self, all_params, features): 
        activations = features 
        for i, (weights, biases) in enumerate(all_params):
            if i == len(all_params) - 1: # Remove sigmoid activation from the last layer (output layer)
                activations = jnp.dot(activations, weights) + biases
            else:
                z = jnp.dot(activations, weights) + biases
                activations = self.activation_function(z)
        return activations

class BathTubPlant:
    def __init__(self, A, C, H_0, lower, upper):
        self.A = A
        self.C = C
        self.target = H_0
        self.H = H_0
        self.g = 9.8
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 

    def update(self, U):
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper)
        V = jnp.sqrt(2 * self.g * self.H) 
        Q = V * self.C 
        change_in_volume = U + D - Q
        change_in_height = change_in_volume / self.A
        self.H += change_in_height
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

    def update(self,U): 
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper) 

        self.q1 += U
        self.q2 += D

        self.q1 = jnp.clip(self.q1,0,1) #Setting q1 to 0 if q1<0, and 1 if q1>0
        self.q2 = jnp.clip(self.q2,0,1) #Setting q2 to 0 if q2<0, and 1 if q2>0

        q = self.q1 + self.q2
        price = self.pmax - q
        P1 = self.q1 * (price - self.c_m)
        return P1   

class HeaterRoomPlant: 
    def __init__(self, initial_temp, target, lower_noise, upper_noise):
        self.temperature = initial_temp  
        self.lower_noise = lower_noise
        self.upper_noise = upper_noise
        self.target = target

    def update(self, U):
        D = np.random.uniform(low=self.lower_noise, high=self.upper_noise)
        delta_temperature = jnp.array(U) / 5.0 + D
        new_temperature = self.temperature + delta_temperature
        self.temperature = new_temperature
        return self.temperature

def main():
    with open("config1.json") as json_data_file:
        config = json.load(json_data_file)

    bathtub = BathTubPlant(config['Bathtub']['A'], config['Bathtub']['C'], config['Bathtub']['H0'], config["Disturbance"]["rangeNoiseLower"], config["Disturbance"]["rangeNoiseLower"])
    cournotPlant = CournotPlant(config["CournotPlant"]["pmax"], config["CournotPlant"]["c_m"],config["CournotPlant"]["T"], config["Disturbance"]["rangeNoiseLower"], config["Disturbance"]["rangeNoiseLower"])
    heater_room_plant = HeaterRoomPlant(config["HeaterRoom"]["initial_temp"],config["HeaterRoom"]["target_temp"],  config["Disturbance"]["rangeNoiseLower"], config["Disturbance"]["rangeNoiseLower"])

    pid_controller = PIDController(config["PID"]["kp"],config["PID"]["ki"], config["PID"]["kd"])
    neural_controller = NeuralController(config["Neural"]["layers"], config["Neural"]["activation_function"],config["Neural"]["lower_p"], config["Neural"]["upper_p"])
    
    consys = CONSYS(neural_controller,heater_room_plant, config["Consys"]["learning_rate"])
    consys.run_system(config["Consys"]["num_epochs"],config["Consys"]["timesteps"])


if __name__ == "__main__":
    main()

#Bathtub:
    #PID: Learning_rate = 1, timesteps = 10
    #Neural: Learning_rate = 0.1, timesteps = 20
#Cournotplant:
    #PID & neural: Learning_Rate = 0.0001, timesteps = 10/20
#Heaterroom:
    # PID/Neural: Learning_rate = 0.01, timesteps = 20