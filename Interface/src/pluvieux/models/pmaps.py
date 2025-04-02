# pylint: disable=cyclic-import
""" Pmap : la carte proprement dite """
from django.db import models
from pluvieux.models import Layer

class Pmap (models.Model):
    """ Aggrégation des couleurs et des layers """
    name = models.CharField(max_length=30, primary_key=True)
    description =  models.CharField(max_length=255)
    layer1 = models.ForeignKey(Layer, related_name="+", on_delete=models.RESTRICT)
    layer2 = models.ForeignKey(Layer, related_name="+", on_delete=models.RESTRICT)
    layer3 = models.ForeignKey(Layer, related_name="+", on_delete=models.RESTRICT)

    class Meta:
        """ Nom de l'app """
        app_label  = 'pluvieux'
    def __str__(self):
        return str(self.description)

# pylint: enable=cyclic-import
