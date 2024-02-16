from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

# Load a model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('input.mp4')
count = 0

# Store processed frames in a list
processed_frames = []

# List to store bounding box coordinates
bounding_boxes = []

while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        
    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x1+x, y1+y), (255, 0, 0), 2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Store bounding box coordinates
                bounding_boxes.append((x, y, x1, y1))
                 
    processed_frames.append(img.copy())

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

bounding_boxes_json = json.dumps(bounding_boxes)

# Write processed frames to a video file
output_path = 'output/output_videos/video_30fps.mp4'
coordinates_path = 'output/coordinates/coordinates_30fps.json'
fps = 30  
height, width, _ = processed_frames[0].shape
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for frame in processed_frames:
    out.write(frame)
# Write the JSON data to the output JSON file
with open(coordinates_path, 'w') as f:
    f.write(bounding_boxes_json)


out.release()

# Print the bounding box coordinates
print("Bounding Box Coordinates:")
for box in bounding_boxes:
    print(box)
