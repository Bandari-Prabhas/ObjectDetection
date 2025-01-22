import time
import mediapipe as mp
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "efficientdet_lite0.tflite"

options = ObjectDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE,
    score_threshold=0.0
)

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (int(bbox.origin_x) + MARGIN, int(bbox.origin_y) - ROW_SIZE)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image

img_path = 'man-bmw.jpg'
img = cv2.imread(img_path)

with ObjectDetector.create_from_options(options) as detector:
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    start_time = time.time()
    detection_result = detector.detect(mp_frame)
    end_time = time.time()

    annotated_frame = visualize(img, detection_result)
    fps = f"FPS: {round(1 / (end_time - start_time), 2)}"
    cv2.putText(annotated_frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2_imshow(annotated_frame)

    cv2.waitKey(0)

cv2.destroyAllWindows()
