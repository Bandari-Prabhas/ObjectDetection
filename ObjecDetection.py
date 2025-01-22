import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

img = cv2.imread('man-bmw.jpg')

ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

object_names = []
for class_id, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    object_names.append(f"{classLabels[class_id - 1]}: {conf:.2f}")
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    label_text = f"{classLabels[class_id - 1]}"
    text_location = (box[0], box[1] - 10)
    cv2.putText(img, label_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

print("Detected Objects:", object_names)
