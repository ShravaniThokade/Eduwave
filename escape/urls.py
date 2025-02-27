from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name="index"),
    path('register', views.register, name="register"),
    path('login', views.login, name="login"),
    path('dashboard', views.dashboard, name="dashboard"),
    path('user_logout', views.user_logout, name="user_logout"),
    path('study_set', views.study_set, name="study_set"),
    path('import_materials', views.import_materials, name='import_materials'),
    path('transcription', views.transcription, name="transcription"),
    path('test', views.test, name="test"),
    path('youtube', views.youtube, name="youtube"),
    path('generate_summary', views.generate_summary, name='generate_summary'),
]