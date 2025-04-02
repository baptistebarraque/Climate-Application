""" Admin magique django """
from django.contrib import admin
from pluvieux.models import Color
from pluvieux.models import DataSource
from pluvieux.models import Pmap
from pluvieux.models import Layer
from pluvieux.models import UserMapSettings

admin.site.register(Color)
admin.site.register(DataSource)
admin.site.register(Pmap)
admin.site.register(Layer)
admin.site.register(UserMapSettings)
