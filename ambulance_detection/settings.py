import os
from pathlib import Path
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ Debug Mode (Make sure it's True for now)
DEBUG = True  

# ✅ Allowed Hosts (For local development & Render deployment)
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

# ✅ Security Settings (Keep secret key hidden)
SECRET_KEY = os.getenv("SECRET_KEY", "your-default-secret-key")

# Installed apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'home',  # ✅ Ensure this matches your app name
]

# Custom User Model
AUTH_USER_MODEL = 'home.CustomUser'

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # ✅ Whitenoise for static files
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ambulance_detection.urls'

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],  # ✅ Ensure templates folder exists
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ambulance_detection.wsgi.application'

# Database (Using SQLite for local development)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ✅ Authentication settings
LOGIN_REDIRECT_URL = 'home:main'  # After login, go to main
LOGIN_URL = 'home:login'  # If not logged in, redirect here
LOGOUT_REDIRECT_URL = 'home:login'  # After logout, go to login

# 🔥 CSRF Fixes 🔥
SESSION_COOKIE_SECURE = False  # Change to True in production
CSRF_COOKIE_SECURE = False  # Change to True in production
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
]

X_FRAME_OPTIONS = 'DENY'  # Prevent clickjacking attacks

# ✅ Static Files (Whitenoise for Render)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "static"]  # Ensure "static" folder exists
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# ✅ Media Files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# ✅ YOLO Model Paths
YOLO_WEIGHTS = os.path.join(BASE_DIR, 'yolo_model', 'yolov3.weights')
YOLO_CONFIG = os.path.join(BASE_DIR, 'yolo_model', 'yolov3.cfg')
AMBULANCE_CLASS_ID = 2  # Adjust according to your YOLO class index

# ✅ Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
