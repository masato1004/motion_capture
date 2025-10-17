# Import the necessary libraries
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from time import time
import torch

# Import from local modules
from camera_controller import CameraController
from pose_estimator import PoseEstimator
import Filter



STREAMING_SUBJECT = 0  # Change to video file path for video input
STREAMING_SUBJECT = '../videos/IMG_8369.mov'  # Change to video file path for video input
FILTER_NAME = 'kalman'  # Options: 'kalman', 'lowpass', 'moving_average'

if __name__ == "__main__":
    # Initialize pose estimator
    pose_estimator = PoseEstimator(version='v11') # 'v11' or 'v8'
    initial_estimation_results = None
    
    filter = Filter.define_filter(FILTER_NAME, num_states=len(pose_estimator.keypoints_vector()), num_measurements=len(pose_estimator.keypoints_vector()))
    
    filters = {key_number: Filter.define_filter(FILTER_NAME, num_states=4, num_measurements=4) for key_number in pose_estimator.current_keypoints.keys()}
    
    # Initialize camera and start capturing
    camera_controller = CameraController(streaming_subject = STREAMING_SUBJECT)
    camera_controller.start_capture()
    
    
    # Initialize keypoints history for filtering
    ret, frame = camera_controller.capture()
    while True:
        ret, frame = camera_controller.capture()
        if not ret or frame is None:
            print("Failed to capture initial image")
            continue
        
        initial_estimation_results = pose_estimator.estimate_person(frame)
        if initial_estimation_results.keypoints.conf.numpy().size == 0:
            print("Initial estimation failed, retrying...")
            continue
        
        if len(initial_estimation_results.keypoints.xy[0]) > 0:
            pose_estimator.set_keypoints(np.array([[xy[0], xy[1]] for i, xy in enumerate(initial_estimation_results.keypoints.xy[0]) if i > 4]).reshape(-1,2), initial_estimation_results.keypoints[0].conf.reshape(-1,1))
            pose_estimator.copy_keypoints()
            break
        else:
            print("Low confidence in initial estimation, retrying...")
    else:
        pose_estimator.set_keypoints(initial_estimation_results.keypoints.xy.numpy() if initial_estimation_results is not None else [], initial_estimation_results.keypoints.conf.numpy() if initial_estimation_results is not None else [])
    filter.x = pose_estimator.keypoints_vector()
    for k, key_number in enumerate(pose_estimator.current_keypoints.keys()):
        filters[key_number].x = pose_estimator.keypoints_vector()[[k*2, k*2+1, k*2+len(pose_estimator.current_keypoints), k*2+1+len(pose_estimator.current_keypoints)],0].reshape(-1,1)
    
    
    # Prepare for saving the output as a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    output_fps = 20  # Frames per second for the output video
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    output_file = 'tmp.mp4'
    video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, (frame_width, frame_height))
    
    k = 0
    loop_time = 0.05  # Initial loop time
    while True:
        loop_start = time()
        
        # ---- multi threaded capture (uncomment to use) ----
        estimate_start = time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            capture_exec = executor.submit(camera_controller.capture)
            estimate_exec = executor.submit(pose_estimator.estimate_person, camera_controller.current_frame if camera_controller.current_frame.size != 0 else np.zeros((480, 640, 3), dtype=np.uint8))
            # estimate_exec = executor.submit(pose_estimator.estimate, camera_controller.current_frame if camera_controller.current_frame.size != 0 else np.zeros((480, 640, 3), dtype=np.uint8))
            ret, frame = capture_exec.result()
            # annotated_frame = estimate_exec.result()

            
            estimation_results = estimate_exec.result()
            
            # filtering
            
            
            # annotated_frame = estimation_results.plot() if estimation_results is not None else np.array([])
            
            
            if not ret or frame is None:
                print("Failed to capture image")
                cv2.destroyAllWindows()
                break
        estimate_end = time()
        estimate_frequency = 1 / (estimate_end - estimate_start)
        
        for cls, conf, box, keypoints, keypoints_conf in zip(estimation_results.boxes.cls, estimation_results.boxes.conf, estimation_results.boxes.xyxy, estimation_results.keypoints.xy, estimation_results.keypoints.conf):
            class_name = estimation_results.names[int(cls)]
            if class_name == "person" and conf > 0.7:
                pose_estimator.set_keypoints(np.array([[xy.numpy()[0], xy.numpy()[1]] for i, xy in enumerate(keypoints) if i > 4]).reshape(-1,2), np.array([keypoints_conf.numpy()[i] for i in range(len(keypoints_conf)) if i > 4]).reshape(-1,1))
                # print(np.array([[xy.numpy()[0], xy.numpy()[1]] for i, xy in enumerate(keypoints) if i > 4]).reshape(-1,1).shape)
                filtered_x = filter.smooth(pose_estimator.keypoints_vector(loop_time),dt=loop_time)
                
                # with ThreadPoolExecutor(max_workers=len(pose_estimator.current_keypoints.keys())) as executor:
                #     filtered_x_series = {key_number: executor.submit(filters[key_number].smooth, pose_estimator.keypoints_vector(loop_time)[[k*2, k*2+1, k*2+len(pose_estimator.current_keypoints), k*2+1+len(pose_estimator.current_keypoints)],0].reshape(-1,1), dt=loop_time, observed=pose_estimator.current_keypoints[key_number]["confidence"]>0.) for k, key_number in enumerate(pose_estimator.current_keypoints.keys())}
                #     new_keypoints = np.array([filtered_x_series[k].result()[:2,0].reshape(1,2) for k in pose_estimator.current_keypoints.keys()]).reshape(-1,2)
                #     pose_estimator.update_keypoints(new_keypoints, np.array([keypoints_conf.numpy()[i] for i in range(len(keypoints_conf)) if i > 4]).reshape(-1,1))
                
                # filtered_x = filter.smooth(np.array([[xy.numpy()[0], xy.numpy()[1]] for i, xy in enumerate(keypoints) if i > 4]).reshape(-1,1), dt=loop_time)
                # print("differential of estimated and filtered:", f"{pose_estimator.keypoints_vector[:int(len(pose_estimator.keypoints_vector)/2)] - filtered_x[:int(len(filtered_x)/2)]}")

                pose_estimator.update_keypoints(np.array([[filtered_x[k], filtered_x[k+1]] for k in range(0,len(pose_estimator.current_keypoints)*2,2)]).reshape(-1,2), np.array([keypoints_conf.numpy()[i] for i in range(len(keypoints_conf)) if i > 4]).reshape(-1,1))
                # pose_estimator.set_keypoints(np.array([[filtered_x[k], filtered_x[k+1]] for k in range(0,len(pose_estimator.current_keypoints)*2,2)]), np.array([keypoints_conf.numpy()[i] for i in range(len(keypoints_conf)) if i > 4]).reshape(-1,1))
                # print("filtered keypoints:", pose_estimator.keypoints_vector[:int(len(filtered_x)/2)].reshape(-1,2))
        
        # # conventional capture
        # ret, frame = camera_controller.capture()
        # if not ret or frame is None:
        #     print("Failed to capture image")
        #     cv2.destroyAllWindows()
        #     break
        
        # # Estimate pose on the captured frame
        # estimate_start = time()
        # annotated_frame = pose_estimator.estimate_person(frame)
        # estimate_end = time()
        # estimate_frequency = 1 / (estimate_end - estimate_start)
        
        # Add frame to history for potential replay
        camera_controller.add_frame_to_history(frame, int(np.round(estimate_frequency)))
        
        # Display the annotated frame
        annotated_frame = pose_estimator.draw_person(frame, estimation_results, [pose_estimator.current_keypoints[i]["coords"] for i in pose_estimator.current_keypoints.keys()])
        if annotated_frame.size != 0:
            cv2.imshow('Pose Estimation', annotated_frame)
            # Write the annotated frame to the output video
            video_writer.write(annotated_frame)

        loop_end = time()
        
        loop_time = loop_end - loop_start
        
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
    
    # Release the video writer
    video_writer.release()
    