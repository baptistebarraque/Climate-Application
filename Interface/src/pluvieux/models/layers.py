""" Modele pour les couches """
from datetime import timedelta, datetime
from django.db import models
from . import DataSource
from . import Color


class Layer (models.Model):
    """ Une carte a trois tuiles """
    # XXX: le titre est malheureux à cause du tuilage cher aux GIS
    name = models.CharField(max_length=30, primary_key=True)
    description = models.CharField(max_length=255)
    data_source = models.ForeignKey(DataSource, on_delete=models.RESTRICT)
    color = models.ForeignKey(Color, on_delete=models.RESTRICT)

    def get_params(self, date):
        """ Retourne ce dont on a besoin pour construire la carte """
        h = {}
        h['color'] = self.color.get_degraded()
        h['collection'] = self.data_source.collection
        h['band'] = self.data_source.band
        h['min'] = self.data_source.minimum
        h['max'] = self.data_source.maximum
        h['opacity'] = self.data_source.inital_opacity
        date = date
        h['date'] = datetime.strptime(date, '%Y-%m-%d')
        last = h['date'] + timedelta(days=self.data_source.interval)
        h['last'] = last.strftime('%Y-%m-%d')
        return h

    class Meta:
        """ Nom de l'app """
        app_label = 'pluvieux'

    def __str__(self):
        return str(self.description)
