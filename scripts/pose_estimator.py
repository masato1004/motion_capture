import cv2
import numpy as np
from ultralytics import YOLO
from time import time

# for suppressing warnings
from logging import getLogger
logger = getLogger('ultralytics')
logger.disabled = True

threshold_person = 0.7
threshold_keypoint = 0.6

class PoseEstimator:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')  # Replace '/path/to/' with the actual path to the model file
        self.__curent_frame = None

    def estimate_person(self, frame):
        # Perform pose estimation
        results = self.model(frame)

        results_0 = results[0]

        # Create a copy of the frame for annotations
        person_annotated_frame = frame.copy()

        # Filter and annotate only "person" detections with confidence > threshold_person
        for cls, conf, box, keypoints, keypoints_conf in zip(results_0.boxes.cls, results_0.boxes.conf, results_0.boxes.xyxy, results_0.keypoints.xy, results_0.keypoints.conf):
            class_name = results_0.names[int(cls)]
            if class_name == "person" and conf > threshold_person:
                # print(f"Detected 'person' with confidence: {conf:.2f}")
                # Draw the bounding box on the frame
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(person_annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(person_annotated_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.draw_stick_model(person_annotated_frame, keypoints, keypoints_conf)
                
                # Apply Kalman filter to smooth keypoints
                # self.linear_kalman_filter(keypoints)
                # for i, (x, y) in enumerate(keypoints):
                #     if keypoints_conf[i] > threshold_keypoint:
                #         cv2.circle(person_annotated_frame, (int(x), int(y)), 5, (0, 200, 100), -1)
                #         cv2.putText(person_annotated_frame, str(i), (int(x)+5, int(y)+5),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1)
                #         print(f"Keypoint {i}: ({x:.2f}, {y:.2f}) with confidence {keypoints_conf[i]:.2f}")

                

        return person_annotated_frame
    
    def linear_kalman_filter(self, points, dt=1/30):
        # Define the Kalman filter parameters
        num_points = points.shape[0]
        state_size = 4
        meas_size = 2
        kf = cv2.KalmanFilter(state_size, meas_size * num_points)
        kf.measurementMatrix = np.zeros((meas_size * num_points, state_size * num_points), np.float32)
        for i in range(num_points):
            kf.measurementMatrix[meas_size*i:meas_size*(i+1), state_size*i:state_size*(i+1)] = np.array([[1, 0, 0, 0],
                                                                                                        [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.eye(state_size * num_points)
        for i in range(num_points):
            kf.transitionMatrix[state_size*i:state_size*(i+1), state_size*i:state_size*(i+1)] = np.array([[1, 0, dt, 0],
                                                                                                          [0, 1, 0, dt],
                                                                                                          [0, 0, 1, 0],
                                                                                                          [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(state_size * num_points) * 1e-2
        kf.measurementNoiseCov = np.eye(meas_size * num_points) * 1e-1
        kf.errorCovPost = np.eye(state_size * num_points)
        # Initialize state
        kf.statePost = np.zeros((state_size * num_points, 1), np.float32)
        for i in range(num_points):
            kf.statePost[state_size*i:state_size*(i+1), 0] = np.array([[points[i, 0]],
                                                                       [points[i, 1]],
                                                                       [0],
                                                                       [0]], np.float32)
        # Predict and correct
        prediction = kf.predict()
        measurement = points.flatten().reshape(-1, 1).astype(np.float32)
        kf.correct(measurement)
        
    def draw_stick_model(self, person_annotated_frame, keypoints, keypoints_conf):
        # Draw stick figure using keypoints
        for i, (x, y) in enumerate(keypoints):
            if keypoints_conf[i] > threshold_keypoint:
                cv2.circle(person_annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        # Define connections (e.g., COCO keypoint pairs)
        connections = [
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        colors = {
            "arms": (0, 0, 255),  # Red for arms
            "torso": (0, 255, 0),  # Green for torso
            "legs": (255, 50, 50)  # Blue for legs
        }
        for start, end in connections:
            if keypoints_conf[start] > threshold_keypoint and keypoints_conf[end] > threshold_keypoint:
                if start < len(keypoints) and end < len(keypoints):
                    x1, y1 = keypoints[start]
                    x2, y2 = keypoints[end]
                    if (start, end) in [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10)]:
                        color = colors["arms"]
                    elif (start, end) in [(5, 11), (6, 12), (11, 12)]:
                        color = colors["torso"]
                    else:
                        color = colors["legs"]
                    cv2.line(person_annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

if __name__ == "__main__":
    pose_estimator = PoseEstimator()
    
    # Initialize camera and
    cap = cv2.VideoCapture(0)
    while True:
        # capture a test image
        ret, frame = cap.read()
        
        if frame is None:
            print("Failed to capture image")
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            start_time = time()
            annotated_frame = pose_estimator.estimate_person(frame)
            end_time = time()
            
            # print(f"Pose estimation took {end_time - start_time:.2f} seconds")
            
            # Display the annotated frame
            cv2.imshow('Pose Estimation', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break