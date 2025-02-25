import torch
import cv2
import statistics

from weights.parser import get_path_model_object, get_path_model_char
from helper import get_char_id_dict

# Load models
modelPlate = torch.hub.load('yolov5', 'custom', get_path_model_object(), source='local', force_reload=True)
modelCharX = torch.hub.load('yolov5', 'custom', get_path_model_char(), source='local', force_reload=True)

def detect_plate_chars(image):
    results_plate = modelPlate(image)
    plates = []
    for *xyxy, conf, _ in results_plate.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        crop_img = image[y1:y2, x1:x2]
        chars, confidences, char_detected = [], [], []

        results = modelCharX(crop_img)
        detections = results.pred[0]
        detections = sorted(detections, key=lambda x: x[0])  # sort by x coordinate
        for det in detections:
            conf = det[4]
            if conf > 0.5:
                cls = det[5].item()
                char = get_char_id_dict().get(str(int(cls)), '')
                chars.append(char)
                confidences.append(conf.item())
                char_detected.append(det.tolist())
        charConfAvg = round(statistics.mean(confidences) * 100) if confidences else 0
        plate = ''.join(chars)
        return plate, (x1, y1, x2, y2)
    return None , None



def process_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Error loading image from path: {image_path}")

    # Convert BGR to RGB (YOLOv5 expects RGB images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plates = detect_plate_chars(image)
    return plates



result = process_image("./assets/test.png")


print(result)
