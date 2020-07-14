# import libraries

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#import jdc

from ipywidgets import interactive
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm

# creating the unscented kalman filter class

class UKF:
    def __init__(self,sensor_sigma, velocity_sigma, position_sigma,dt):
        # Sensor measurment covariance
        self.r_t = np.array([[sensor_sigma**2]])

        # Motion model for velocity and position
        self.q_t = np.array([[velocity_sigma**2,0.0],
                            [0.0, position_sigma**2]])

        self.dt = dt

        self.mu = np.array([[0.0],
                            [0.0]])

        self.sigma = np.array([[0.0,0.0],
                                [0.0,0.0]])

        self.mu_bar = self.mu
        self.sigma_bar = self.sigma

        self.n = self.q_t.shape[0]
        self.sigma_points = np.zeros((self.n,2*self.n +1))

        # Creating the contestants 
        self.alpha = 1.0
        self.betta = 2.0
        self.k = 3.0 - self.n
        
        self.lam = self.alpha**2 * (self.n + self.k) - self.n
        self.gamma = np.sqrt(self.n + self.lam)
        
        self.x_bar = self.sigma_points
        
        self.a = np.array([[1.0,0.0],
                        [self.dt, 1.0]])
        self.b = np.array([[self.dt],
                            [0.0]])
        
    def initial_values(self,mu_0, sigma_0):
        self.mu = mu_0
        self.sigma = sigma_0
    
    def compute_sigmas(self):
        S = sqrtm(self.sigma)
        # TODO: Implement the sigma points 
        self.sigma_points[:, 0] = self.mu[:, 0]
    
        self.sigma_points[:, 1] = self.mu[:, 0]
        self.sigma_points[:, 2] = self.mu[:, 0]
        self.sigma_points[:, 1:3] += self.gamma * S
    
        self.sigma_points[:, 3] = self.mu[:, 0]
        self.sigma_points[:, 4] = self.mu[:, 0]
        self.sigma_points[:, 3:5] -= self.gamma * S
    
        return self.sigma_points

    def g(self,u):
        g = np.zeros((self.n, self.n+1))
        g = np.matmul(self.a, self.sigma_points) + self.b * u
        return g
   
    def predict(self, u):
        # TODO: Implement the predicting step
        self.compute_sigmas()
        x_bar = self.g(u)
    
        self.x_bar = x_bar
        return x_bar

    def h(self,Z):
        return np.matmul(np.array([[0.0, 1.0]]), Z) 
    
    def weights_mean(self):
    
        w_m = np.zeros((2*self.n+1, 1))
        # TODO: Calculate the weight to calculate the mean based on the predicted sigma points
    
        w_m[0] = self.lam/(self.n + self.lam) 
        w_m[1] = 1.0/(self.n + self.lam)/2
        w_m[2] = 1.0/(self.n + self.lam)/2
        w_m[3] = 1.0/(self.n + self.lam)/2
        w_m[4] = 1.0/(self.n + self.lam)/2
    
        self.w_m = w_m
        return w_m

    def weights_cov(self):
    
        w_cov = np.zeros((2*self.n+1, 1))
            # TODO: Calculate the weight to calculate the covariance based on the predicted sigma points
    
        w_cov[0] = self.lam/(self.n + self.lam) + 1.0 - self.alpha**2 + self.betta
        w_cov[1] = 1.0/(self.n + self.lam)/2
        w_cov[2] = 1.0/(self.n + self.lam)/2
        w_cov[3] = 1.0/(self.n + self.lam)/2
        w_cov[4] = 1.0/(self.n + self.lam)/2
    
        self.w_cov = w_cov
        return w_cov

    def update(self,z_in):
        
        # TODO: Implement the update step 
        mu_bar = np.matmul(self.x_bar, self.weights_mean())             # Line 8
        cov_bar=np.matmul(self.x_bar-mu_bar,np.transpose(self.x_bar-mu_bar) * self.weights_cov()) + self.q_t # Line 9
        z = self.h(self.x_bar)                                        # Line 10
        mu_z = np.matmul(z, self.weights_mean())                        # Line 11 
        cov_z = np.matmul(z - mu_z, np.transpose(z - mu_z) * self.weights_cov()) + self.r_t # Line 12 
        cov_xz = np.matmul(self.x_bar - mu_bar, np.transpose(z - mu_z) * self.weights_cov())  # Line 13
        k = np.matmul(cov_xz, np.linalg.inv(cov_z))                   # Line 14
    
        mu_t =  mu_bar  + k * (z_in - mu_z)                           # Line 15
        cov_t = cov_bar - np.matmul(k, cov_z*np.transpose(k))         # Line 16
    
        self.mu = mu_t
        self.sigma = cov_t
    
        return mu_t, cov_t

# initialization 

z = 2.0                         # Initial position
v = 1.0                         # Initial velocity
dt = 1.0                        # The time difference between measures
motion_error = 0.01             # Sensor sigma
velocity_sigma = 0.01           # Velocity uncertainty
position_sigma = 0.01           # Position uncertainty


mu_0 = np.array([[v],
                 [z]]) 

cov_0 = np.array([[velocity_sigma**2, 0.0],
                    [0.0, position_sigma**2]])

u = np.array([0.0])     # no commant is given \ddot{z} = 0 

MYUKF=UKF(motion_error, velocity_sigma, position_sigma, dt)

MYUKF.initial_values(mu_0, cov_0)

MYUKF.initial_values(mu_0, cov_0)

u = 0 # no control input is given
print(MYUKF.predict(0))

z_measured = 3.11
print(MYUKF.update(z_measured))


