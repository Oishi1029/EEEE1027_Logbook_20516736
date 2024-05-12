#https://www.youtube.com/watch?v=7EmfDY2Ooqo&t=229s

from ultralytics import YOLO
import cv2
import supervision as sv
import onnx
from onnx2torch import convert
from picamera2 import Picamera2   
import time 
picam2 = Picamera2()   
picam2.preview_configuration.main.size = (1280,720)   
picam2.preview_configuration.main.format = "RGB888"   
picam2.preview_configuration.align()   
picam2.configure("preview")   
picam2.start() 
time.sleep(2) 
# Path to your ONNX model
onnx_model_path = '/path/to/your/model.onnx'

# Convert the ONNX model to PyTorch
torch_model = convert(onnx_model_path)

model = YOLO(torch_model)

model.set_classes("arrow")

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv2.LabelAnnotator()

while True:
    img= picam2.capture_array()
    
    results = model.predict(img)
    
    detection = sv.Detections.from_ultralytics(results[0])
    
    annotated_frame = bounding_box_annotator.annotate(
        scene = img.copy()
        detections=detections
    )
    
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    out.write(annotated_frame)
    cv2.imshow("Image",annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

