import cv2
import numpy as np
from time import time

class CameraController:
    def __init__(self, streaming_subject=0):
        self.streaming_subject = streaming_subject
        self.cap = None
        self.is_capturing = False
        self._frame_history = []
        self._current_frame = np.array([])
        
    def start_capture(self):
        if self.is_capturing:
            print("Capture is already running.")
            return
        self.cap = cv2.VideoCapture(self.streaming_subject)
        if not self.cap.isOpened():
            raise ValueError("Could not open the camera.")
        self.is_capturing = True
            
    def capture(self):        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to capture image")
            return ret, None
        self._current_frame = frame.copy()
        return ret, frame
    
    def add_frame_to_history(self, frame, frequency=15):
        if len(self._frame_history) >= frequency*4:
            self._frame_history = self._frame_history[-frequency*4:]
        self._frame_history.append(frame)

    def stop_capture(self):
        if not self.is_capturing:
            print("Capture is not running.")
            return
        self.is_capturing = False
        # Release everything when done
        self.cap.release()
        # out.release()
    
    @property
    def current_frame(self):
        return self._current_frame
    
    @property
    def frame_history(self):
        return self._frame_history

if __name__ == "__main__":
    controller = CameraController()
    controller.start_capture()
    k = 0
    while True:
        loop_start = time()
        frame = controller.capture()
        if frame is None:
            break
        # Display the frame
        cv2.imshow('Video Capture', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        loop_end = time()
        
        # Print the frequency of the loop
        if k % 30 == 0:
            print(f"Loop frequency: {1/(loop_end - loop_start):.2f} Hz")
        k += 1
        
        # last_time = loop_end - loop_start