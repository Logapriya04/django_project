import os
import cv2
import torch
import numpy as np
import threading
from pathlib import Path
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from django.contrib.staticfiles import finders
from playsound import playsound # type: ignore
from .models import CustomUser
from .yolo_utils import load_yolo_model


# ✅ Home Page
def index(request):
    return render(request, "home/index.html")

# ✅ Other Basic Views
def main(request):
    return render(request, "home/main.html")

def about(request):
    return render(request, "home/about.html")

def how_it_works(request):
    return render(request, "home/how_it_works.html")

def contact(request):
    return render(request, "home/contact.html")

def service(request):
    return render(request, "home/service.html")  # Ensure "service.html" exists

# ✅ Help Page (Chatbot)
def help_view(request):
    return render(request, "home/help.html")  # Ensure help.html exists in templates/home/

# ✅ Detection Page
def detection_page(request):
    return render(request, "home/detection.html")  # Ensure detection.html exists


model = load_yolo_model()

# ✅ Load YOLO Model
MODEL_PATH = Path(settings.BASE_DIR) / "home" / "yolo_model.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(MODEL_PATH), force_reload=True)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {e}")

# ✅ Sound Alert Function
def play_alert_sound():
    sound_path = finders.find("sounds/alarm.mp3")
    if sound_path:
        threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
    else:
        print("❌ Alert sound file not found!")

# ✅ Authentication
@csrf_exempt
def register(request):
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '').strip()
        password2 = request.POST.get('password2', '').strip()

        if not all([username, email, password1, password2]):
            return JsonResponse({"error": "All fields are required."}, status=400)

        if password1 != password2:
            return JsonResponse({"error": "Passwords do not match!"}, status=400)

        if CustomUser.objects.filter(username=username).exists():
            return JsonResponse({"error": "Username already exists!"}, status=400)

        if CustomUser.objects.filter(email=email).exists():
            return JsonResponse({"error": "Email is already registered!"}, status=400)

        try:
            user = CustomUser.objects.create_user(username=username, email=email, password=password1)
            user.save()
        except Exception as e:
            return JsonResponse({"error": f"Error creating user: {e}"}, status=500)

        user = authenticate(request, username=username, password=password1)
        if user:
            login(request, user)
            return JsonResponse({"message": "✅ Registration successful!"})
        else:
            return JsonResponse({"error": "Authentication failed. Please log in manually."}, status=400)

    return JsonResponse({"error": "Invalid request method."}, status=400)

@csrf_exempt
def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            return JsonResponse({"error": "Username and password are required."}, status=400)

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return JsonResponse({"message": "✅ Login successful!"})
        else:
            return JsonResponse({"error": "❌ Invalid username or password."}, status=400)

    return JsonResponse({"error": "Invalid request method."}, status=400)

@login_required(login_url='home:login')
def user_logout(request):
    logout(request)
    return JsonResponse({"message": "✅ You have successfully logged out."})


@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video = request.FILES["video"]
        file_path = os.path.join(settings.MEDIA_ROOT, "videos", video.name)

        with default_storage.open(file_path, "wb") as dest:
            for chunk in video.chunks():
                dest.write(chunk)

        return JsonResponse({"message": "✅ Video uploaded successfully!", "file_path": file_path})

    return JsonResponse({"error": "Invalid request"}, status=400)


# ✅ Upload Image API
@csrf_exempt
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        file_path = os.path.join(settings.MEDIA_ROOT, "uploads", image.name)

        with default_storage.open(file_path, "wb") as dest:
            for chunk in image.chunks():
                dest.write(chunk)

        return JsonResponse({"message": "✅ Image uploaded successfully!", "file_path": file_path})

    return JsonResponse({"error": "Invalid request"}, status=400)

# ✅ Detect Ambulance API
@csrf_exempt
def detect_ambulance(request):
    if request.method == "POST" and request.FILES.get("file"):
        try:
            file = request.FILES["file"]
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            model = load_yolo_model()
            if model is None:
                return JsonResponse({"error": "YOLO model failed to load."}, status=500)

            results = model(image_rgb)
            detections = results.pandas().xyxy[0]
            detected = False

            for _, row in detections.iterrows():
                label = row["name"]
                confidence = row["confidence"]

                if label.lower() == "ambulance" and confidence > 0.3:
                    detected = True
                    x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(image, "Ambulance Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_filename = "detected_output.jpg"
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
            cv2.imwrite(output_path, image)

            if detected:
                play_alert_sound()
                return JsonResponse({
                    "detected": True,
                    "message": "🚑 Ambulance detected! Alert triggered.",
                    "output_image": settings.MEDIA_URL + output_filename
                })
            else:
                return JsonResponse({"detected": False, "message": "No ambulance detected."})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid Request"}, status=400)

# ✅ CCTV Streaming
def generate_frames():
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = load_yolo_model()
            if model:
                results = model(image_rgb)
                for *xyxy, conf, cls in results.xyxy[0]:
                    label = model.names[int(cls)]
                    if label == "ambulance":
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        play_alert_sound()

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

def cctv_stream(request):
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

def yolo_detection(request):
    return JsonResponse({"message": "🚀 YOLO detection endpoint is working!"})