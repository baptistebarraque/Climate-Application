"""
URL configuration for pluvieux project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('maps/', include('maps.urls'))
"""
from django.urls import path,include
from django.contrib import admin
from .views import cartes
from .views import load_map_settings
from .views import save_map_settings
from .views import get_all_map_settings
from .views import delete_map_settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', cartes, name='cartes'),
    path('accounts/', include('accounts.urls')),
    path('load_map_settings/', load_map_settings, name='load_map_settings'),
    path('save_map_settings/', save_map_settings, name='save_map_settings'),
    path('get_all_map_settings/', get_all_map_settings, name='get_all_map_settings'),
    path('delete_map_settings/', delete_map_settings, name='delete_map_settings'),
]

      
