import numpy as np
import cv2
from scipy.signal import butter, lfilter
from collections import deque

def allowed_filters():
    return filters

class KalmanFilter():
    def __init__(self, num_states, num_measurements):
        self.num_states = num_states
        self.num_measurements = num_measurements
        
        # State vector
        self.x = np.zeros((num_states, 1)).reshape(-1,1)
        
        # State covariance matrix
        self.P = np.ones((num_states,num_states)) * 3
        
        # Measurement matrix
        self.H = np.hstack((np.eye(num_measurements), np.zeros((num_measurements, num_states - num_measurements))))
        
        # Measurement noise covariance
        self.R = np.eye(num_measurements) * 250
        
        # Process noise covariance
        self.Q = np.eye(num_states) * 5
        
        # Kalman gain
        # self.K = np.zeros((self.num_states, self.num_measurements))
        
    @property
    def K(self):
        # Kalman gain
        self._K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        return self._K
    
    def A(self, dt=0.05):
        # State transition Matrix
        _A = np.eye(self.num_states)
        for i in range(int(self.num_states/2)):
            _A[i, i + int(self.num_states/2)] = dt
        return _A
            
    def state_equation(self, x, dt=0.05):
        x_new = self.A(dt) @ x
        return x_new
        
    def predict_step(self, dt=0.05):
        self.x = self.state_equation(self.x, dt).reshape(-1,1)
        self.P = self.A(dt) @ self.P @ self.A(dt).T + self.Q
        return self.x
        
    def filtering_step(self, z):
        # print('z',z)
        # print('x',self.x)
        y = z - self.H @ self.x  # Measurement residual
        self.x = self.x + self.K @ y
        self.P = (np.eye(self.num_states) - self.K @ self.H) @ self.P
        return self.x
    
    def smooth(self, z, dt=0.05, observed=True):
        # print('before',self.x)
        self.predict_step(dt)
        # print('before2',self.x)
        if observed:
            self.filtering_step(z)
        # print(z-self.x[:24])
        # print('after',self.x)
        return self.x

class LowPassFilter():
    def __init__(self, cutoff=0.1, fs=30, order=5):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, 1))
        
    def apply(self, data):
        filtered_data, self.zi = lfilter(self.b, self.a, data, axis=0, zi=self.zi)
        return filtered_data
    
class MovingAverageFilter():
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def apply(self, data):
        self.buffer.append(data)
        return np.mean(self.buffer, axis=0)

filters = {
    "kalman": KalmanFilter,
    "lowpass": LowPassFilter,
    "moving_average": MovingAverageFilter
    }

if __name__ == "__main__":
    print("here")