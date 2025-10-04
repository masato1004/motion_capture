import numpy as np
import cv2
from scipy.signal import butter, lfilter
from collections import deque

class KalmanFilter():
    def __init__(self, num_states, num_measurements):
        self.num_states = num_states
        self.num_measurements = num_measurements
        
        # State vector
        self.x = np.zeros((num_states, 1))
        
        # State covariance matrix
        self.P = np.eye(num_states)
        
        # Measurement matrix
        self.H = np.zeros((num_measurements, num_states))
        
        # Measurement noise covariance
        self.R = np.eye(num_measurements) * 3
        
        # Process noise covariance
        self.Q = np.eye(num_states)
        
        # Kalman gain
        self.K = np.zeros((self.num_states, self.num_measurements))
        
    @property
    def K(self):
        # Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
    
    @property
    def A(self, dt=0.05):
        # State transition Matrix
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def state_equation(self, x, dt=0.05):
        x_new = self.A(dt) @ x
        return x_new
        
    def predict_step(self, x, dt=0.05):
        self.x = self.state_equation(self.x, dt)
        self.P = self.A(dt) @ self.P @ self.A(dt).T + self.Q
        return self.x
        
    def filtering_step(self, z):
        y = z - self.H @ self.x
        self.x = self.x + self.K @ (y - self.H @ self.x)
        self.P = (np.eye(self.num_states) - self.K @ self.H) @ self.P
        return self.x