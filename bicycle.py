import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Class to detect (e.g., bicycle, which is label 2 in COCO dataset)
class_to_detect = 'bicycle'

# Initialize color for bounding box and text
color = (255, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Path to the image file to detect objects
image_path = r'D:\python\xulyanh\word-image-18.png.jpeg'

# Read the image from the path
img = cv2.imread(image_path)

# Use the model to detect objects in the image
results = model(img)

# Get the list of labels from the model
labels = results.names

# Iterate through the detected objects and draw bounding boxes for bicycles
for detection in results.pred[0]:
    # Convert the tensor index to integer
    idx = int(detection[-1].item())

    if labels[idx] == class_to_detect:
        conf = detection[4]
        box = detection[:4].tolist()
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_to_detect} {conf:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), font, 0.5, color, 2)

# Show the image with detected bicycles
cv2.imshow("YOLOv5 Bicycle Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
