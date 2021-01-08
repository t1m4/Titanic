from django.contrib import admin
from django.urls import path, include

from core import views

urlpatterns = [
    path('start/', views.StartView.as_view(), name='core-start'),
]
