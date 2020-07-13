

# importing library 
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from StateSpaceDisplay import state_space_display, state_space_display_updated

# -------------------creating Kalman Filter Class-----------------------------
class KF:
    def __init__(self, measurement_noise,
                        process_noise_v,
                        process_noise_p,
                        dt):
        # process noise
        self.q_t = np.array([[process_noise_v**2,0.0],
                            [0.0,process_noise_p**2]]
                            )
        # measurement noise
        self.r_t = np.array([measurement_noise**2])

        self.dt = dt

        # default velocity and position

        self.mu = np.array([[0.0],[0.0]])

        # default covariance- completely certianty
        self.sigma = np.array([[0.0,0.0],[0.0,0.0]])

        self.mu_bar = self.mu
        self.sigma_bar = self.sigma

        self.a = np.array([[1.0,0.0],[self.dt,1.0]])
        self.b = np.array([[self.dt],[0.0]])

    def g(self,mu,u):
        return np.matmul(self.a,mu)+ self.b * u
    def g_prime(self):
        return self.a

    # setting initial state of the drone
    def initial_values (self,mu_0,sigma_0):
        self.mu = mu_0
        self.sigma = sigma_0

    # prediction step
    def predict(self,u):
        mu_bar = self.g(self.mu,u)
        g_prime = self.g_prime()
        sigma_bar = np.matmul(g_prime,np.matmul(self.sigma,np.transpose(g_prime)))+ self.q_t

        self.mu_bar = mu_bar
        self.sigma_bar = sigma_bar
        return mu_bar, sigma_bar

    # update step (sensor)
    def h(self,mu):
        return np.matmul(np.array([[0.0,1.0]]),mu)
    def h_prime(self):
        return np.array([[0.0,1.0]])
    def update(self,z):
        H = self.h_prime()
        S = np.matmul(np.matmul(H,self.sigma_bar),np.transpose(H)) + self.r_t
        # kalman gain
        K = np.matmul(np.matmul(self.sigma_bar, np.transpose(H)), np.linalg.inv(S))
        mu = self.mu_bar + np.matmul(K, (z - self.h(self.mu_bar)))
        sigma = np.matmul((np.identity(2) - np.matmul(K, H)), self.sigma_bar)
        self.mu = mu
        self.sigma = sigma
    
        return mu, sigma

print("Kalman Filter class is created")
#----------------Initializatoin of KF-------------------

measurement_noise = 0.1
process_noise_v = 0.1
process_noise_p = 0.1

v = 1.0
z = 0.0

dt = 1.0

velocity_sigma = 0.1
position_sigma = 0.1

mu_0 = np.array([[v],
                 [z]]) 

sigma_0 = np.array([[velocity_sigma**2, 0.0],
                    [0.0, position_sigma**2]])

u = np.array([0.0])  
measurement = 1.01

print("Initialization is completed")

# ------------Initialize the object--------------
my_KF = KF(measurement_noise,process_noise_v,process_noise_p,dt)
print("Kalman Filter object is created")

# ----------------Input the initial values --------------------
my_KF.initial_values(mu_0,sigma_0)
# Call the predict function
mu_bar, sigma_bar = my_KF.predict(u)
 
print("Prediction step is done")
print("mu_bar \n")
print(mu_bar)
print("sigma_bar", sigma_bar)
# plot 
state_space_display(z, v, mu_0, sigma_0, mu_bar, sigma_bar)

#------------------Updating step ------------------------------
mu_updated, sigma_updated = my_KF.update(measurement)

print("Update Mean: ")
print(mu_updated)
print("Update Sigma: ")
print(sigma_updated)

state_space_display_updated(z, v, mu_0, sigma_0, mu_bar, sigma_bar, mu_updated, sigma_updated)