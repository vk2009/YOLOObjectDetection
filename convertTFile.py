# from ultralytics import YOLO
# # model: /Users/viditkwatra/Robotics_Antares_Vision_2025/AlgaeDetection/AlgaeDetectionModel/algaeDetectionBest.pt
# # Load your PyTorch model
# model = YOLO('/Users/viditkwatra/Robotics_Antares_Vision_2025/AlgaeDetection/AlgaeDetectionModel/algaeDetectionBest.pt')
# model.export(format="tflite", int8=True)

from ultralytics import YOLO

# # Load the YOLO11 model
# model = YOLO("/Users/viditkwatra/Robotics_Antares_Vision_2025/AlgaeDetection/AlgaeDetectionModel/algaeDetectionBest.pt")

# # Export the model to TFLite format
# model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("algaeDetectionBest_float16.tflite")

# Run inference
results = tflite_model("https://ultralytics.com/images/bus.jpg")