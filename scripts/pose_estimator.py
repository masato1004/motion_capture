import cv2
import numpy as np
import copy
from ultralytics import YOLO
from time import time

# for suppressing warnings
from logging import getLogger
logger = getLogger('ultralytics')
logger.disabled = True

THRESHOLD_PERSON = 0.7
THRESHOLD_KEYPOINT = 0.2
YOLO_VERSION = 'v8'  # 'v11' or 'v8'

class PoseEstimator:
    def __init__(self, version=YOLO_VERSION):
        if version == 'v11':
            self.model = YOLO('yolo11n-pose.pt')
        elif version == 'v10':
            self.model = YOLO('yolov10n.pt')
        elif version == 'v9':
            self.model = YOLO('yolov9c.pt')
        elif version == 'v8':
            self.model = YOLO('yolov8n-pose.pt')  # Replace '/path/to/' with the actual path to the model file
        self._curent_frame = np.array([])
        self._curent_annotated_frame = np.array([])
        self._keypoints = {
            5:  {"part": 'left_shoulder', "coords": np.zeros(2), "confidence": 0.0},
            6:  {"part": 'right_shoulder', "coords": np.zeros(2), "confidence": 0.0},
            7:  {"part": 'left_elbow', "coords": np.zeros(2), "confidence": 0.0},
            8:  {"part": 'right_elbow', "coords": np.zeros(2), "confidence": 0.0},
            9:  {"part": 'left_wrist', "coords": np.zeros(2), "confidence": 0.0},
            10: {"part": 'right_wrist', "coords": np.zeros(2), "confidence": 0.0},
            11: {"part": 'left_hip', "coords": np.zeros(2), "confidence": 0.0},
            12: {"part": 'right_hip', "coords": np.zeros(2), "confidence": 0.0},
            13: {"part": 'left_knee', "coords": np.zeros(2), "confidence": 0.0},
            14: {"part": 'right_knee', "coords": np.zeros(2), "confidence": 0.0},
            15: {"part": 'left_ankle', "coords": np.zeros(2), "confidence": 0.0},
            16: {"part": 'right_ankle', "coords": np.zeros(2), "confidence": 0.0}
        }
        self._privious_keypoints = copy.deepcopy(self._keypoints)
    
    def set_keypoints(self, keypoints, conf):
        [self._keypoints[i+5].update({"coords": keypoints[i,:], "confidence": conf[i]}) for i in range(len(keypoints))]
        
    def copy_keypoints(self):
        self._privious_keypoints = copy.deepcopy(self._keypoints)
    
    def estimate(self, frame) -> np.ndarray:
        if frame is None:
            return self._curent_annotated_frame
        
        # Store the current frame
        self._curent_frame = frame.copy()
        
        # Perform pose estimation
        results = self.model(frame)

        results_0 = results[0]

        # Create a copy of the frame for annotations
        annotated_frame = results_0.plot()
                
        self._curent_annotated_frame = annotated_frame
        return annotated_frame
        

    def estimate_person(self, frame) -> np.ndarray:
        if frame is None:
            return self._curent_annotated_frame
        
        # Store the current frame
        self._curent_frame = frame.copy()
        
        # Perform pose estimation
        # print(frame.shape)
        results = self.model(frame)

        results_0 = results[0]
        
        # if len(results_0.keypoints.xy) != 0:
        #     [self._keypoints[i].update({"coords": results_0.keypoints.xy[0].numpy()[0,i,:], "confidence": results_0.keypoints.conf[0].numpy()[0,i]}) for i in self._keypoints.keys()]
        
        # # Create a copy of the frame for annotations
        # person_annotated_frame = frame.copy()
        # for cls, conf, box, keypoints, keypoints_conf in zip(results_0.boxes.cls, results_0.boxes.conf, results_0.boxes.xyxy, results_0.keypoints.xy, results_0.keypoints.conf):
        #     class_name = results_0.names[int(cls)]
        #     if class_name == "person" and conf > THRESHOLD_PERSON:
        #         self.draw_stick_model(person_annotated_frame, keypoints, keypoints_conf)
        
        return results_0

    def draw_person(self, frame, results_0, filtered_keypoints) -> np.ndarray:
        person_annotated_frame = frame.copy()
        # Filter and annotate only "person" detections with confidence > THRESHOLD_PERSON
        for cls, conf, box, keypoints, keypoints_conf in zip(results_0.boxes.cls, results_0.boxes.conf, results_0.boxes.xyxy, results_0.keypoints.xy, results_0.keypoints.conf):
            class_name = results_0.names[int(cls)]
            if class_name == "person" and conf > THRESHOLD_PERSON:
                # print(f"Detected 'person' with confidence: {conf:.2f}")
                # Draw the bounding box on the frame
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(person_annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add background for text
                text = f"{class_name} {conf:.2f}"
                font_scale = 2
                thickness = 5
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x, text_y = x1, y1 - 10
                cv2.rectangle(person_annotated_frame, (text_x, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)

                # Draw the text
                cv2.putText(person_annotated_frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                
                self.draw_stick_model(person_annotated_frame, keypoints, keypoints_conf, filtered_keypoints)
                
                # Apply Kalman filter to smooth keypoints
                # self.linear_kalman_filter(keypoints)
                # for i, (x, y) in enumerate(keypoints):
                #     if keypoints_conf[i] > THRESHOLD_KEYPOINT:
                #         cv2.circle(person_annotated_frame, (int(x), int(y)), 5, (0, 200, 100), -1)
                #         cv2.putText(person_annotated_frame, str(i), (int(x)+5, int(y)+5),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1)
                #         print(f"Keypoint {i}: ({x:.2f}, {y:.2f}) with confidence {keypoints_conf[i]:.2f}")

                
        self._curent_annotated_frame = person_annotated_frame
        return person_annotated_frame
    
    def linear_kalman_filter(self, point, dt=1/30):
        # Initialize Kalman filter parameters
        if not hasattr(self, '_kalman'):
            self._kalman = cv2.KalmanFilter(4, 2)
            self._kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]], np.float32)
            self._kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                  [0, 1, 0, dt],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], np.float32)
            self._kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], np.float32) * 0.03

        # Update Kalman filter with the new measurement
        measurement = np.array([[np.float32(point[0])],
                     [np.float32(point[1])]])
        self._kalman.correct(measurement)

        # Predict the next position
        prediction = self._kalman.predict()
        return int(prediction[0]), int(prediction[1])

    def draw_stick_model(self, person_annotated_frame, keypoints, keypoints_conf, filtered_keypoints=np.array([])):
        # Draw stick figure using keypoints
        for i, (x,y) in enumerate(keypoints):
                
            if i > 4 and len(filtered_keypoints) > 0:
                # print(i, np.array([x, y]) , np.array([filtered_keypoints[i-5][0][0], filtered_keypoints[i-5][1][0]]))
                # x, y = filtered_keypoints[i-5][0][0], filtered_keypoints[i-5][1][0]
                # if self.current_keypoints[i]["confidence"] > THRESHOLD_KEYPOINT:
                #     print(self.current_keypoints[i]['part'], np.array([x, y]) , np.array([filtered_keypoints[i-5][0], filtered_keypoints[i-5][1]]))
                cv2.circle(person_annotated_frame, (int(self.current_keypoints[i]["coords"][0]), int(self.current_keypoints[i]["coords"][1])), 10, (250, 100, 250), -1)
            
            # if i > 4 and len(filtered_keypoints) > 0:
            #     x, y = self.current_keypoints[i]["coords"]
            #     cv2.circle(person_annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)
            # elif keypoints_conf[i] > THRESHOLD_KEYPOINT:
                # cv2.circle(person_annotated_frame, (int(x), int(y)), 8, (0, 0, 255), -1)
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
        alpha = 0.7  # Transparency factor
        for start, end in connections:
            if len(filtered_keypoints) > 0:
                x1, y1 = self.current_keypoints[start]["coords"]
                x2, y2 = self.current_keypoints[end]["coords"]
                line_width = 8
                if not (keypoints_conf[start] > 0.5 and keypoints_conf[end] > 0.5):
                    # color = (255, 255, 255)
                    line_width = 8
                elif (start, end) in [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10)]:
                    color = colors["arms"]
                elif (start, end) in [(5, 11), (6, 12), (11, 12)]:
                    color = colors["torso"]
                else:
                    color = colors["legs"]
                cv2.line(person_annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
            # else:
            # if keypoints_conf[start] > THRESHOLD_KEYPOINT and keypoints_conf[end] > THRESHOLD_KEYPOINT:
            #     if start < len(keypoints) and end < len(keypoints):
            #         x1, y1 = keypoints[start]
            #         x2, y2 = keypoints[end]
            #         if (start, end) in [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10)]:
            #             color = tuple(int(colors["arms"][-1-k_]) for k_ in [0,1,2])
            #         elif (start, end) in [(5, 11), (6, 12), (11, 12)]:
            #             color = tuple(int(colors["torso"][-1-k_]) for k_ in [0,1,2])
            #         else:
            #             color = tuple(int(colors["legs"][-1-k_]) for k_ in [0,1,2])
            #         cv2.line(person_annotated_frame, (int(x1)+300, int(y1)), (int(x2)+300, int(y2)), color, 6)

    def update_keypoints(self, new_keypoints, new_conf):
        # print(self._keypoints)
        self._privious_keypoints = copy.deepcopy(self._keypoints)
        # print(self._keypoints)
        # print(self._privious_keypoints)
        # [self._keypoints[i].update({"coords": new_keypoints[i], "confidence": new_conf[i]}) for i in self._keypoints.keys()]
        [self._keypoints[i+5].update({"coords": new_keypoints[i,:], "confidence": new_conf[i]}) for i in range(len(new_keypoints))]
        # print(self._keypoints)
        # print(self._privious_keypoints)
        
    @property
    def current_frame(self):
        return self._curent_frame
    
    @property
    def current_keypoints(self):
        return self._keypoints
    
    def keypoints_vector(self, dt=0.05):
        _keypoints = np.array([
            [self._keypoints[i]["coords"][0], self._keypoints[i]["coords"][1]] for i in self._keypoints.keys()
        ])
        _d_keypoints = self.derrivative_keypoints_vector(dt)
        return np.concatenate([_keypoints.flatten(), _d_keypoints.flatten()]).reshape(-1,1)
        
    def derrivative_keypoints_vector(self, dt=0.05):
        _derrivative_keypoints_vector = np.array([
            [self._keypoints[i]["coords"][0] - self._privious_keypoints[i]["coords"][0],
             self._keypoints[i]["coords"][1] - self._privious_keypoints[i]["coords"][1]]
            for i in self._keypoints.keys()
        ])/dt
        # print("_derrivative_keypoints_vector:", _derrivative_keypoints_vector.flatten())
        return _derrivative_keypoints_vector
        

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