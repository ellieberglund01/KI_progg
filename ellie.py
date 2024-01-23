import jax
import jax.numpy as jnp
import numpy as np


class PIDController:
    def __init__(self, dt, plant):
        self.dt = dt
        self.plant = plant
  
        self.prev_error = 0.0
        self.integral = 0.0
        self.error_history = []

        #if plant == "bathtub" --> set plant parameters


    def run_one_epoch(self, kp, ki, kd, target, initial):

        if self.plant == 'bathtub':
            plant = bathTub(10,8,1)
        
        error = target-initial
        derivative = error
        integral = 0

        for t in (1,self.dt):
            u = kp * error + ki * self.integral + kd * derivative

            #update error
            integral += error
            derivative = error-self.prev_error
            self.prev_error = error
            h = bathTub.get_H
            plant.update_H(u,h)
            error = plant.get_H() - plant.H_0
            self.error_history.append(error)
        
        return integral
       
     
    def update_params(self, learning_rate, gradients):
        self.kp = kp - learning_rate*gradients[0] # Error
        self.ki = ki - learning_rate*gradients[1] # Error history
        self.kd = kd - learning_rate*gradients[2]

    def __call__(self, num_epochs, kp, kd, ki, initial, target, learning_rate):
        mse_history = []
        gradfunc = jax.grad(self.run_one_epoch, argnums=[0,1,2])
        #Init params and state
        for i in range(num_epochs):
            self.error_history = []  # Reset error history for a new epoch
            bathTub.H = initial # Reset plant state to initial state
            gradients = gradfunc(kp,ki,kd, target, initial)
            self.update_params(learning_rate, gradients)
            mse = np.mean(np.square(self.error_history))
            mse_history.append(mse)
    

class CONSYS:
    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant
        self.error_history = []

class bathTub:
    def __init__(self, A, C, H_0):
        self.A = A
        self.C = C
        self.H = H_0
        self.H_0 = H_0
        self.g = 9.8
        self.noise_range = [-0.1,0.1]

    def get_V(self, h):
        V = jnp.sqrt(2 * self.g * h) # Volume
        return V
    
    def get_Q(self, h):
        Q = self.velocity(h) * self.C #flow rate
        return Q

    def update_H(self, u, h):
        D = np.random.uniform(low=self.noise_range[0], high=self.noise_range[1])
        change_in_volume = u + D - self.get_Q(h)
        change_in_height = change_in_volume / self.A
        self.H += change_in_height

    def get_H(self): #Eller er d get change vi burde ha?
        return self.H


dt = 10  # Example timestep
target = 8.0  # Example target height
initial = 5.0  # Example initial height
learning_rate = 0.01
num_epochs = 10
kp=0.1
kd=0.1
ki=0.1


bath = bathTub(10,8,1) 
controller = PIDController(dt, 'bathtub')
controller(num_epochs,kp,kd,ki,initial,target,learning_rate)