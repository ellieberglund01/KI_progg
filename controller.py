import numpy as np
import jax
import jax.numpy as jnp
import math

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
            self.controller.eror_history = jnp.zeros(timesteps)
            mse, gradients = gradfunc(self.controller.get_params(), timesteps)
            print("gradients", gradients)
            mse_history.append(mse)
            self.controller.update_params(self.learning_rate, gradients)
        print(f"Epoch: {epoch + 1}, Average Error: {mse}")  

        return mse_history

    def run_one_epoch(self, params, timesteps): 
        U = 0
        for t in range(timesteps):
            plant_output = self.plant.update_H(U)
            error = self.plant.H_0 - plant_output 
            U = self.controller.compute(params, error, t) 
            print("Hei")
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
    def __init__(self, A, C, H_0):
        self.A = A
        self.C = C
        self.H = H_0
        self.H_0=H_0
        self.g = 9.8
        self.V = jnp.sqrt(2 * self.g * self.H) 
        self.noise_range = [-0.1,0.1] 

    def reset(self):
        self.__init__(self.A,self.C,self.H_0)

    def update_H(self, U):
        D = np.random.uniform(low=self.noise_range[0], high=self.noise_range[1]) #Lage før alle timesteps disuturbance vector lengde timesteps
        Q = self.V * self.C #flow rate
        change_in_volume = U + D - Q
        change_in_height = change_in_volume / self.A
        self.H += change_in_height
        return self.H
    
    def get_H(self):
        return self.H

 
num_epochs = 100
learning_rate = 0.1
timesteps = 100
bathtub = BathTubPlant(A=50,C=5,H_0=10)
pid_controller = PIDController(kp=0.5,ki=0.5,kd=0.1, timesteps=timesteps)
consys = CONSYS(controller=pid_controller,plant=bathtub,learning_rate=learning_rate) 
consys.run_system(num_epochs,timesteps)