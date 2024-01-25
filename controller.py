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
        self.k1_changes = []  # Store changes in k1 across epochs
        self.k2_changes = []  # Store changes in k2 across epochs
        self.k3_changes = []  # Store changes in k3 across epochs

    def run_system(self, num_epochs, timesteps):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) #Ide: argnums=0 kaller på params som er liste med parametere
        mse_history = []
        prev_params = self.controller.get_params()  # Store previous parameters for computing changes
        for epoch in range (1,num_epochs):
            self.plant.reset()
            self.controller.error_history = jnp.zeros(timesteps)
            mse, gradients = gradfunc(self.controller.get_params(), timesteps)
            print("gradients", gradients)
            mse_history.append(mse)
            self.controller.update_params(self.learning_rate, gradients)
            current_params = self.controller.get_params()
            # Compute changes in k1, k2, k3 and store them
            self.k1_changes.append(current_params[0] - prev_params[0])
            self.k2_changes.append(current_params[1] - prev_params[1])
            self.k3_changes.append(current_params[2] - prev_params[2])
            prev_params = current_params
        print(f"Epoch: {epoch + 1}, Average Error: {mse}")  

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
        plt.plot(range(1, num_epochs), self.k1_changes, marker='o', label='Δk1')
        plt.plot(range(1, num_epochs), self.k2_changes, marker='o', label='Δk2')
        plt.plot(range(1, num_epochs), self.k3_changes, marker='o', label='Δk3')
        plt.title('Changes to k parameters across epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Change in k values')
        plt.legend()
        plt.grid(True)
        plt.show()

        return mse_history

    def run_one_epoch(self, params, timesteps): 
        U = 0
        for t in range(timesteps):

            if isinstance(self.plant,BathTubPlant):
                plant_output = self.plant.update_H(U) #Passer til Bathtube
                error = self.plant.H_0 - plant_output  #Passer til Bathtube 

            if isinstance(self.plant,CournotPlant):
                plant_output = self.plant.update_p(U) #Passer til plant 2
                error = plant_output #Passer til plant 2
                
                
            U = self.controller.compute(params, error, t) 

            print("Timestep", t, ":" )
        mse = jnp.mean(jnp.square(self.controller.error_history))
        return mse

class PIDController:
    def __init__(self, kp, ki, kd, timesteps):
        self.params = jnp.array([kp,ki,kd])
        self.error_history = jnp.zeros(timesteps)
        
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

    def update_H(self, U):
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
    def __init__(self, pmax, c_m, T, lower, upper):
        self.pmax = pmax
        self.c_m = c_m
        self.T = T
        self.q1 = 10
        self.q2 = 20
        self.rangeNoiseLower = lower 
        self.rangeNoiseUpper = upper 
        
    def reset(self):
        self.__init__(self.pmax,self.c_m,self.T, self.rangeNoiseLower, self.rangeNoiseUpper)

    def update_p(self,U): #Heller sette Disturbance i controlSys og ha som input her?
        D = np.random.uniform(low=self.rangeNoiseLower, high=self.rangeNoiseUpper) 
        # Update quantities
        self.q1 += U
        self.q2 += D

        self.q1 = min(max(U, 0), 1)  # enforce constraints
        self.q2 = min(max(D, 0), 1)  # enforce constraints


        # Calculate total quantity and price
        q = self.q1 + self.q2
        price = self.pmax - q

        # Calculate profit for producer 1
        P1 = self.q1 * (price - self.c_m)

        # Calculate error
        E = self.T - P1

        return E




kp = 0.5
ki = 0.5
kd = 0.1

num_epochs = 10
learning_rate = 0.1
timesteps = 20
rangeNoiseLower = -0.01
rangeNoiseUpper = 0.01

A=50
C=5
H0=10

pmax = 1000
c_m = 0.1 
T = 200 

bathtub = BathTubPlant(A=A,C=C,H_0=H0, lower=rangeNoiseLower, upper=rangeNoiseUpper)
cournotPlant = CournotPlant( pmax=pmax, c_m =c_m, T=T, lower=rangeNoiseLower, upper=rangeNoiseUpper)
 
pid_controller = PIDController(kp=kp,ki=ki,kd=ki, timesteps=timesteps)


consys = CONSYS(controller=pid_controller,plant=cournotPlant,learning_rate=learning_rate)
consys.run_system(num_epochs,timesteps)