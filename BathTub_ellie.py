import numpy as np
import matplotlib.pyplot as plt

#Goal is to minimize change in height 
class BathTub:
    def __init__(self, A, C, H_0):
        self.A = A
        self.C = C
        self.H = H_0
        self.g = 9.8
        self.history = {'time': [], 'height': []}

    def update_H(self, U, D): #U is controller output, D is random noise #Dt er per timestep
        V = jnp.sqrt(2 * self.g * self.H_0) # Volume
        Q = velocity * drain_area #flow rate
        change_in_volume = U + D - Q 
        change_in_height = change_in_volume / self.A
        self.H += change_in_height

    def get_H(self): #Eller er d get change vi burde ha?
        return self.H