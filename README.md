# Estimation


## Kalman Filter with PID for Coaxial Drone


## Unscented Kalman Filter for Coaxial Drone
you will become familiar with the UKF method which is a robust tool for estimating the value of the measured quantity. Later we will apply it to estimate the position of the one-dimensional quadcopter with can move only in the vertical axis. 

Next, we create the class that will have all the functions needed to perform the localization of the object in the one-dimensional environment. 

For simplicity, will use a drone that can only move in the vertical direction for the given drone the state function is simply vertical position and velocity $X=(\dot{z},z)$. The control input for the drone is the vertical acceleration $u = \ddot{z}$. For KF we have to define the measurement error associated with the measuring the hight variable. 
