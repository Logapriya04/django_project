import torch
from pathlib import Path
from django.conf import settings

# Define the model path
MODEL_PATH = Path(settings.BASE_DIR) / "home" / "yolo_model.pt"

def load_yolo_model():
    global model
    if "model" not in globals():
        try:
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(MODEL_PATH),
                force_reload=False
            )
            model.eval()
            print("✅ YOLO Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            model = None
    return model
