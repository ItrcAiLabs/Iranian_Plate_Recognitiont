import cv2
import numpy as np
from weights.parser import get_path_model_object, get_path_model_char
from ultralytics import YOLO
import torch
import os

# Check if the model paths exist
print(os.path.exists(get_path_model_object()))
print(os.path.exists(get_path_model_char()))

class PlateRecognizer:
    def __init__(self, conf_threshold: float = 0.4) -> None:
        # Load the models for plate and character recognition
        self.model_plate = self.load_model(get_path_model_object())
        self.model_char = self.load_model(get_path_model_char())
        self.conf_threshold = conf_threshold
        # Character class names (add more if needed)
        self.charclassnames = [
            '0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', 
            '1', 'malul', 'n', 's', 'sad', 't', 'ta', 'v', 'y', '2', '3', 
            '4', '5', '6', '7', '8'
        ]
        
    def load_model(self, model_path: str):
        """Load and return the YOLO model."""
        return YOLO(model_path)

    def recognize_plate(self, image_path: str) -> tuple:
        """Recognize the plate number in an image."""
        # Load the image
        image = cv2.imread(image_path)
        
        # Run plate detection
        result = self.model_plate.predict(image, conf=self.conf_threshold)

        for detected in result:
            bbox = detected.boxes
            for box in bbox:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                cls_name = int(box.cls[0])  # Get predicted class
                
                # Only process if the detected object is a plate (class 1)
                if cls_name == 1:
                    plate_img = image[y1:y2, x1:x2]  # Extract the plate region
                    
                    # Run character recognition on the extracted plate image
                    plate_output = self.model_char(plate_img, conf=self.conf_threshold)
                    bbox = plate_output[0].boxes.xyxy
                    cls = plate_output[0].boxes.cls

                    # Extract characters and their positions
                    char_keys = cls.numpy().astype(int)
                    char_values = bbox[:, 0].numpy().astype(int)
                    
                    # Sort characters by position to get correct order
                    sorted_chars = sorted(zip(char_keys, char_values), key=lambda x: x[1])
                    
                    # Build the recognized plate string
                    plate_number = ''.join(self.charclassnames[i[0]] for i in sorted_chars)
                    return plate_number, (x1, y1, x2, y2)
        
        # Return None if no plate is detected
        return None, None

