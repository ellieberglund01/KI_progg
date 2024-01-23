import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class CONSYS:
    def __init__(self, controller, plant,learning_rate):
        self.controller = controller
        self.plant = plant
        self.learning_rate = learning_rate
        self.params = controller.get_params()


    def run_system(self, num_epochs, timesteps):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) #Ide: argnums=0 kaller på params som er liste med parametere
        mse_history = []
        for epoch in range (1,num_epochs):
            self.plant.reset()
            self.controller.error_history = jnp.zeros(timesteps)
            mse, gradients = gradfunc(self.controller.get_params(), timesteps)
            #print("gradients", gradients)
            mse_history.append(mse)
            self.controller.update_params(self.learning_rate, gradients)
       
        #print(f"Epoch: {epoch + 1}, Average Error: {mse}")  
        self.controller.plot(mse_history)
        return mse_history

    def run_one_epoch(self, params, timesteps): 
        U = 0
        for t in range(timesteps):
            plant_output = self.plant.update(U) 
            error = self.controller.target - plant_output          
            U = self.controller.compute(params, error, t) 
            #print("Timestep", t, ":" )
        mse = jnp.mean(jnp.square(self.controller.error_history))
        return mse
    
    

class PIDController:
    def __init__(self, kp, ki, kd, timesteps, target):
        self.params = jnp.array([kp,ki,kd])
        self.error_history = jnp.zeros(timesteps)
        self.target = target
        self.k1_his = []  # Store changes in k1 across epochs
        self.k2_his = []  # Store changes in k2 across epochs
        self.k3_his = []  # Store changes in k3 across epochs
        
    def get_params(self):
        return self.params
    
    def compute(self,params, error,timestep):
        kp = params[0]
        ki = params[1]
        kd = params[2]
        self.error_history = self.error_history.at[timestep].set(error)
        p = kp*error
        d = kd*(error-self.error_history[timestep-1])
        i = ki*jnp.sum(self.error_history)
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


class BathTubPlant:
    def __init__(self, A, C, H_0, lower, upper):
        self.A = A
        self.C = C
        self.H = H_0
        self.H_0=H_0
        self.g = 9.8
        self.noise_range = [-0.1,0.1] 
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 


    def reset(self):
        self.__init__(self.A,self.C,self.H_0, self.rangeNoiseLower, self.rangeNoiseUpper)

    def update(self, U):
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper) #Lage før alle timesteps disturbance vector lengde timesteps, mer naturlig der enn her?
        V = jnp.sqrt(2 * self.g * self.H) 
        Q= V * self.C 
        change_in_volume = U + D - Q
        change_in_height = change_in_volume / self.A
        self.H += change_in_height
        return self.H
    
    def get_H(self):
        return self.H


class CournotPlant:
    def __init__(self, pmax, c_m, lower, upper):
        self.pmax = pmax
        self.c_m = c_m
        self.q1 = 0.5
        self.q2 = 0.5
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 
        
    def reset(self):
        self.__init__(self.pmax,self.c_m,self.rangeNoiseLower, self.rangeNoiseUpper)

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
class HeaterRoomPlant:
    def __init__(self, initial_temp, lower_noise, upper_noise):
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
T = 30

#Bathtub
A=100
C=1
H0=1

#Cournotplant
pmax = 10
c_m = 0.1 

#Heaterroom
initial_temp = 20

bathtub = BathTubPlant(A=A,C=C,H_0=H0, lower=rangeNoiseLower, upper=rangeNoiseUpper)
cournotPlant = CournotPlant( pmax=pmax, c_m =c_m, lower=rangeNoiseLower, upper=rangeNoiseUpper)
heater_room_plant = HeaterRoomPlant(initial_temp, rangeNoiseLower, rangeNoiseUpper)
 
pid_controller = PIDController(kp=kp,ki=ki,kd=ki,target=T, timesteps=timesteps)


consys = CONSYS(controller=pid_controller,plant=heater_room_plant,learning_rate=learning_rate)
consys.run_system(num_epochs,timesteps)