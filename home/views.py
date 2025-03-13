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
from playsound import playsound  # type: ignore
from .models import CustomUser
from ultralytics import YOLO


# ✅ Define YOLO Model Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")  # Model should be in the root directory

# ✅ Ensure model exists, otherwise download it
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at {MODEL_PATH}, downloading...")
    os.system(f"wget -O {MODEL_PATH} https://github.com/ultralytics/assets/releases/download/v8/yolov8n.pt")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Failed to download YOLO model at {MODEL_PATH}")

# ✅ Load YOLO Model
print(f"✅ Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)


# ✅ Home Page
def index(request):
    return render(request, "home/index.html")


# ✅ Other Pages
def main(request):
    return render(request, "home/main.html")


def about(request):
    return render(request, "home/about.html")


def how_it_works(request):
    return render(request, "home/how_it_works.html")


def contact(request):
    return render(request, "home/contact.html")


def service(request):
    return render(request, "home/service.html")


# ✅ Help Page (Chatbot)
def help_view(request):
    return render(request, "home/help.html")


# ✅ Detection Page
def detection_page(request):
    return render(request, "home/detection.html")


# ✅ Play Alert Sound
def play_alert_sound():
    sound_path = finders.find("sounds/alarm.mp3")
    if sound_path:
        threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
    else:
        print("❌ Alert sound file not found!")


# ✅ Register User
@csrf_exempt
def register(request):
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '').strip()
        password2 = request.POST.get('password2', '').strip()

        if not all([username, email, password1, password2]):
            messages.error(request, "❌ All fields are required.")
            return redirect("home:register")

        if password1 != password2:
            messages.error(request, "❌ Passwords do not match!")
            return redirect("home:register")

        if CustomUser.objects.filter(username=username).exists():
            messages.error(request, "❌ Username already exists!")
            return redirect("home:register")

        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, "❌ Email is already registered!")
            return redirect("home:register")

        try:
            user = CustomUser.objects.create_user(username=username, email=email, password=password1)
            user.save()
            messages.success(request, "✅ Registration successful! Please log in.")
            return redirect("home:login")
        except Exception as e:
            messages.error(request, f"❌ Error creating user: {e}")
            return redirect("home:register")

    # Render the registration page if it's a GET request
    return render(request, "home/register.html")


# ✅ Login User
@csrf_exempt
def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            messages.error(request, "❌ Username and password are required.")
            return redirect("home:login")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            messages.success(request, "✅ Login successful!")
            return redirect("home:main")  # Redirect to main page
        else:
            messages.error(request, "❌ Invalid username or password.")
            return redirect("home:login")

    # If GET request, render the login page
    return render(request, "home/login.html")


# User Logout
def user_logout(request):
    logout(request)
    messages.success(request, "You have successfully logged out.")
    return redirect("home:index")  # Redirect to the home page


# ✅ Detect Ambulance in Image
@csrf_exempt
def detect_ambulance(request):
    if request.method == "POST" and request.FILES.get("file"):
        try:
            file = request.FILES["file"]
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = model(image_rgb)
            detections = results[0].boxes.data.cpu().numpy()  # Fixing detection output
            detected = False

            for box in detections:
                x1, y1, x2, y2, conf, cls = box
                label = model.names[int(cls)]

                if label.lower() == "ambulance" and conf > 0.3:
                    detected = True
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(image, "Ambulance Detected", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
            results = model(image_rgb)

            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                label = model.names[int(cls)]
                if label == "ambulance":
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    play_alert_sound()

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


def cctv_stream(request):
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")
