""" main form to request a map """
from django import forms
from pluvieux.models import Pmap


class MapForm(forms.Form):
    """ the form """
    mymap = forms.ModelChoiceField(
        label="Choisir une carte",
        queryset=Pmap.objects.all(),
        required=True,
    )
    date = forms.DateField(
        label="Choisir une date",
        widget=forms.DateInput(attrs={'type': 'date'}),
        required=True,
    )
    adress = forms.CharField(label="Zone d'intérêt (Rentrer une adresse)", max_length=255)
