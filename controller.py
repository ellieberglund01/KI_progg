import numpy as np
import jax
import jax.numpy as jnp

class CONSYS:
    def __init__(self, controller, plant,learning_rate):
        self.controller = controller
        self.plant = plant
        self.learning_rate = learning_rate
        self.params = controller.get_params()

    def run_system(self, num_epochs, timesteps):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) #Ide: argnums=0 kaller på params som er liste med parametere
        mse_history = []
        for i in range (1,num_epochs):
            mse, gradients = gradfunc(self.controller.get_params(), timesteps)
            mse_history.append(mse)
            self.controller.update_params(self.learning_rate, gradients)
    
        return mse_history

    def run_one_epoch(self, params, timesteps): #Vi bruker ikke params, det er et problem
        error_history = []
        error = self.plant.get_H() - self.plant.H_0
        self.plant.reset()
        U = 0
        for t in range(timesteps):
            plant_output = self.plant.update_H(U)
            error = self.plant.H_0 - plant_output 
            error_history.append(error)
            U = self.controller.compute(error)
        mse = jnp.mean(jnp.array(error_history)**2)
        return mse

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = jax.numpy.array(kp)
        self.ki = jax.numpy.array(ki)
        self.kd = jax.numpy.array(kd)
        self.prev_error = 0
        self.integral = 0
        
    def get_params(self):
        params = [self.kp, self.ki, self.kd]
        return params
    
    def compute(self, error):
        self.integral += error
        derivative = error-self.prev_error #derivative?
        U = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return U
    
    def update_params(self,learning_rate, gradients):
        self.kp = self.kp - learning_rate*gradients[0] # Error
        self.ki = self.ki - learning_rate*gradients[1] # Error history
        self.kd = self.kd - learning_rate*gradients[2]

class BathTubPlant:
    def __init__(self, A, C, H_0):
        self.A = A
        self.C = C
        self.H = H_0
        self.H_0=H_0
        self.g = 9.8
        self.noise_range = [-0.1,0.1] # Denne vet jeg ikke hvordan skal se ut

    def reset(self):
        self.__init__(self.A,self.C,self.H_0)

    def update_H(self, U):
        D = np.random.uniform(low=self.noise_range[0], high=self.noise_range[1])
        V = jnp.sqrt(2 * self.g * self.H) # Velocity
        Q = V * self.C #flow rate
        change_in_volume = U + D - Q
        change_in_height = change_in_volume / self.A
        self.H += change_in_height
        return self.H
    
    def get_H(self):
        return self.H

 
num_epochs = 100
learning_rate = 0.1
timesteps = 100
bathtub = BathTubPlant(A=50,C=50,H_0=10)
pid_controller = PIDController(kp=0.5,ki=0.5,kd=0.1)
consys = CONSYS(controller=pid_controller,plant=bathtub,learning_rate=learning_rate) 
consys.run_system(num_epochs,timesteps)