import numpy as np
import jax
import jax.numpy as jnp


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = jax.numpy.array(kp)
        self.ki = jax.numpy.array(ki)
        self.kd = jax.numpy.array(kd)
        self.prev_error = jnp.zeros_like(self.kp)
        self.integral = jnp.zeros_like(self.kp)

    def compute(self, error):
        self.integral += error
        U = self.kp * error + self.ki * self.integral + self.kd * (error - self.prev_error) 
        self.prev_error = error
        return U

    def update_parameters(learningRate, gradients):
        self.kp -= learning_rate * gradients[0]
        self.ki -= learning_rate * gradients[1]
        self.kd -= learning_rate * gradients[2]

    def get_params(self):
        return [self.kp, self.ki, self.kd]  # Return PID parameters as a single flat array

class CONSYS:
    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant

    def run_system(self, num_epochs, learningRate, timesteps):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) #Hvordan funker denne?
        mse_history = []
        for epoch in range(num_epochs):
            avg_error,gradients = gradfunc(self.controller.get_params(), timesteps) #Og denne?
            mse_history.append(avg_error)
            self.controller.update_parameters(learningRate, gradients)
        print(f"Epoch: {epoch + 1}, Average Error: {avg_error}")

        return mse_history

    def run_one_epoch(self,params,timesteps): #Hva gjør vi med params?
        error_history = []
        #self.plant.reset()
    
        U = 0 #Blir dette riktig? 

        for t in range(timesteps):
            plant_output = self.plant.update( U) #Endre
            setpoint = self.plant.get_setpoint()
            error = setpoint - plant_output
            error_history.append(error)
            U = self.controller.compute(error)
      
        avg_error = jnp.mean(jnp.array(error_history)**2)
        return avg_error

class BathtubPlant:
    def __init__(self, A, C, H0):
        self.A = A  # Cross-sectional area of the bathtub
        self.C = C  # Cross-sectional area of the drain (typically a small fraction of A)
        self.H0 = H0  # Initial water height in the bathtub
        self.H = H0
        self.noise_range = [-0.1,0.1] # Denne vet jeg ikke hvordan skal se ut

    def reset(self):
        self.__init__(self.A,self.C,self.H_0)

    def update(self, U):
        D = np.random.uniform(low=self.noise_range[0], high=self.noise_range[1])

        # Calculate the velocity of water exiting through the drain
        V = jnp.sqrt(2 * 9.8 * self.H)
        # Calculate the flow rate of exiting water
        Q = V * self.C
        # Update the bathtub volume
        dB_dt = U + D - Q
        # Update the water height
        dH_dt = dB_dt / self.A
        # Update the water height in the bathtub
        self.H += dH_dt
        return self.H
    
    def get_setpoint(self):
        return self.H0
    

num_epochs = 100
learning_rate = 0.1
timesteps = 100

bathtub_plant = BathtubPlant(50, 50, 10)  # Replace with actual values
pid_controller = PIDController(0.5, 0.2, 0.1)  # Example PID gains
con_sys = CONSYS(controller=pid_controller, plant=bathtub_plant)
con_sys.run_system(num_epochs, learning_rate, timesteps)


