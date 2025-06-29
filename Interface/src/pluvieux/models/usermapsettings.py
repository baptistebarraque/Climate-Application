from django.db import models
from django.contrib.auth.models import User

class UserMapSettings(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    location = models.JSONField(default=list)  # Stocker [latitude, longitude]
    zoom = models.IntegerField(default=3)      # Niveau de zoom
    date = models.DateField(null=True, blank=True)  # Date sélectionnée
    layer1_opacity = models.FloatField(default=1.0)  # Opacité de la couche 1
    layer2_opacity = models.FloatField(default=1.0)
    layer3_opacity = models.FloatField(default=1.0)

    class Meta:
        unique_together = ('user', 'name')  # Assurez-vous que chaque utilisateur a des noms uniques pour les paramètres

    def __str__(self):
        return f"Paramètres de carte pour {self.user.username} ({self.name})"
    

class UserPredictionSettings(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    zone_prediction_name=models.CharField(max_length=100)
    zone_prediction_location=models.JSONField(default=list)
    data_file = models.FileField(upload_to='data/', null=True, blank=True)
    model_file = models.FileField(upload_to='models/', null=True, blank=True)

    class Meta:
        unique_together= ('user', 'zone_prediction_name')

