from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("gesture.pt")

# Export the model to NCNN format
model.export(format="ncnn",imgsz=320,half=True)  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
# ncnn_model = YOLO("yolo11n_ncnn_model")

# # Run inference
# for i in range(10):
#     results = ncnn_model.predict("https://ultralytics.com/images/bus.jpg",imgsz=320)