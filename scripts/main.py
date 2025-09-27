# Import the necessary libraries
import cv2
import numpy as np
from time import time

# Import from local modules
from camera_controller import CameraController
from pose_estimator import PoseEstimator

if __name__ == "__main__":
    # Initialize pose estimator
    pose_estimator = PoseEstimator()
    
    # Initialize camera and start capturing
    camera_controller = CameraController(camera_index=0)
    camera_controller.start_capture()
    
    k = 0
    while True:
        loop_start = time()
        ret, frame = camera_controller.capture()
        if not ret or frame is None:
            print("Failed to capture image")
            cv2.destroyAllWindows()
            break
        
        # Estimate pose on the captured frame
        estimate_start = time()
        annotated_frame = pose_estimator.estimate_person(frame)
        estimate_end = time()
        estimate_frequency = 1 / (estimate_end - estimate_start)
        
        # Add frame to history for potential replay
        camera_controller.add_frame_to_history(frame, int(np.round(estimate_frequency)))
        
        # Display the annotated frame
        cv2.imshow('Pose Estimation', annotated_frame)

        loop_end = time()
        
        # Print the frequency of the loop
        if k % 30 == 0:
            print(f"Loop frequency: {1/(loop_end - loop_start):.2f} Hz")
        k += 1
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('r'):
            # Replay the last few seconds
            for past_frame in camera_controller.frame_history:
                annotated_past_frame = pose_estimator.estimate_person(past_frame)
                cv2.imshow('Pose Estimation', annotated_past_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Stop capturing from the camera
    camera_controller.stop_capture()