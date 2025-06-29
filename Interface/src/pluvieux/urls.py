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
from .views import _prediction_interface_view
from .views import load_data_view
from .views import get_user_zones
from .views import train_model_view
from .views import predict_if_rainy_yesterday_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', cartes, name='cartes'),
    path('accounts/', include('accounts.urls')),
    path('load_map_settings/', load_map_settings, name='load_map_settings'),
    path('save_map_settings/', save_map_settings, name='save_map_settings'),
    path('get_all_map_settings/', get_all_map_settings, name='get_all_map_settings'),
    path('delete_map_settings/', delete_map_settings, name='delete_map_settings'),
    path('PredictiveModel/', _prediction_interface_view, name='predictive_model_interface'),
    path('load-data/', load_data_view, name='load_data'),
    path('get-zones/', get_user_zones, name='get_user_zones'),
    path('model-training/', train_model_view, name='train_model_view'),
    path('prediction/', predict_if_rainy_yesterday_view, name='predict_if_rainy_yesterday_view')
]

      
