""" Modele pour les layers """
from django.db import models

class DataSource (models.Model):
    """ couche géographique """
    #XXX: getion des dates
    name =  models.CharField(max_length=30, primary_key=True)
    description =  models.CharField(max_length=255)
    minimum = models.FloatField()
    maximum = models.FloatField()
    interval = models.IntegerField(default=1)
    inital_opacity = models.FloatField(default=0.3)
    collection = models.CharField(max_length=255)
    band =  models.CharField(max_length=255)
    class Meta:
        """ Nom de l'app """
        app_label  = 'pluvieux'
