import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov8s.pt") 
name = "man-bmw.jpg"
img = cv2.imread(name)

results = model.predict(source=img, conf=0.5) 

detected_objects = []
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0] 
    confidence = box.conf[0] 
    class_id = int(box.cls[0]) 
    label = model.names[class_id] 

    if confidence >= 0.55:
        detected_objects.append(label)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if detected_objects:
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
else:
    print("No objects detected with confidence >= 55%.")

if detected_objects:
    print("Detected Objects:", detected_objects)
