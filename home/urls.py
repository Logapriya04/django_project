from django.urls import path
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'home'  # Ensure this matches the app name

urlpatterns = [
    # ✅ Admin Panel
    path('admin/', admin.site.urls),

    # ✅ Home and Main Pages
    path('', views.index, name='index'),
    path('main/', views.main, name='main'),

    # ✅ Information Pages
    path('about/', views.about, name='about'),
    path('how-it-works/', views.how_it_works, name='how_it_works'),
    path('service/', views.service, name='service'),
    path('contact/', views.contact, name='contact'),

    # ✅ Authentication
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),

    # ✅ Help Page (Chatbot)
    path('help/', views.help_view, name='help'),

    # ✅ Detection Page & APIs
    path("detection/", views.detection_page, name="detection"),
    path("detect-ambulance/", views.detect_ambulance, name="detect_ambulance"),

    # ✅ CCTV Streaming
    path("cctv-stream/", views.cctv_stream, name="cctv_stream"),
]

# ✅ Serve static and media files in development mode
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
