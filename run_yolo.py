import os
import cv2
from ultralytics import YOLO

# Create outputs folder if it doesn't exist (exist_ok=True prevents error if already present)
os.makedirs("outputs", exist_ok=True)

# Load pretrained YOLOv8 nano model — auto-downloads yolov8n.pt on first run
model = YOLO("yolov8n.pt")

# Ask user to type or paste the full image path
image_path = input("Enter path to your image: ").strip('"')

# Run object detection; returns a list of Results objects
results = model(image_path)

# Draw bounding boxes, labels and confidence scores on the image
annotated = results[0].plot()

# Save the annotated image inside the outputs folder
output_path = os.path.join("outputs", "result.jpg")
cv2.imwrite(output_path, annotated)
print(f"Saved: {output_path}")

# Loop through every detected box and print class name + confidence
for box in results[0].boxes:
    name = results[0].names[int(box.cls)]   # class label (e.g., "person", "car")
    conf = float(box.conf)                  # confidence score (0.0 – 1.0)
    print(f"- {name}: {conf:.2f}")

# Display the result in an OpenCV window until any key is pressed
cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)          # 0 = wait indefinitely for a key press
cv2.destroyAllWindows() # Close all OpenCV windows
