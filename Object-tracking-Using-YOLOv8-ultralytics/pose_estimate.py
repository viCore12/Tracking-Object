from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLOv8 model for pose estimation
model = YOLO("models/yolov8n-pose.pt")

# Process the image
results = model.predict("test_videos/LONG3297.jpg", show_boxes=False, show_labels=False, show_conf=False, save=True) #Ẩn label video thì truyền trực tiếp

pose_annotated_frame = results[0].plot(labels=False, conf=False, boxes=False) #Ẩn label của ảnh thì truyền vào plot function

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk