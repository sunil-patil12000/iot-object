import torch
from PIL import Image

class ObjectDetector:
    def __init__(self):
        # Load a pre-trained YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
    def detect(self, image_path):
        # Load image
        img = Image.open(image_path)
        
        # Perform inference
        results = self.model(img)
        
        # Process results
        objects_detected = []
        for pred in results.xyxy[0]:  # Process predictions
            x1, y1, x2, y2, conf, cls = pred.tolist()
            label = results.names[int(cls)]
            confidence = float(conf)
            
            if confidence > 0.5:  # Only include objects with >50% confidence
                objects_detected.append({
                    'label': label,
                    'confidence': round(confidence, 2),
                    'bbox': [round(x1), round(y1), round(x2), round(y2)]
                })
                
        return objects_detected
