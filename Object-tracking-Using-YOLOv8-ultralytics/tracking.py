import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 models
model = YOLO("models/yolov8n.pt")
model_pe = YOLO("models/yolov8n-pose.pt")

# Open the video file
video_path = "test_videos/1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
output_path = "output_videos/1_output.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize the variables for FPS calculation
prev_time = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Tracking Object
        results = model.track(frame, persist=True, classes=[0], show=False)  # 0: person
        annotated_frame = results[0].plot()  # Visualize the results on the frame

        # Pose Estimation
        results_pe = model_pe.predict(frame)
        pose_annotated_frame = results_pe[0].plot(labels=False, conf=False, boxes=False)

        # Overlay the pose_annotated_frame onto the original annotated_frame
        combined_frame = cv2.addWeighted(annotated_frame, 0.5, pose_annotated_frame, 0.5, 0)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Add FPS text to the combined frame
        cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(combined_frame)

        # Display the combined frame with FPS
        cv2.imshow("YOLOv8 Tracking and Pose Estimation", combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

