from ultralytics import YOLO

# Load your PyTorch model
model = YOLO('/path/model.pt')

# Export the model to TFLite format
model.export(format="tflite", int8=True) # int8 can be set to false

# Load the exported TFLite model
tflite_model = YOLO("algaeDetectionBest_float16.tflite")

# Run inference
results = tflite_model("https://ultralytics.com/images/bus.jpg")
