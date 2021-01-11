from django.contrib import admin
from django.urls import path, include

from core import views

urlpatterns = [
    path('', views.FormView.as_view(), name='core-form'),
    path('result/', views.ResultView.as_view(), name='core-result'),
    path('start/', views.StartView.as_view(), name='core-start'),
]
