import cv2

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

classLabels = []
with open('labels.txt', 'r') as f:
    classLabels = f.read().strip().split('\n')

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

video_source = 0
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

conf_threshold = 0.5

print("Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Exiting.")
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=conf_threshold)

    if len(ClassIndex) > 0:
        for class_id, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 0 < class_id <= len(classLabels):
                label = classLabels[class_id - 1]
                if label == "cell phone":
                    print(f"Class: {label}, Confidence: {conf:.2f}, BBox: {box}")
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"Invalid class_id: {class_id}. Skipping.")

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
