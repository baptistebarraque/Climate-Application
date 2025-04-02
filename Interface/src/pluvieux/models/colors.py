""" Modele pour les couleurs """
import re
from django.db import models
from django.core.exceptions import ValidationError

def _validate_steps(step):
    """ Les steps sont un entier positif multiple de deux """
    if step > 0 and step % 2 == 0:
        return step
    raise ValidationError("step must be a positive, pair, non null integer")

def _validate_color(color):
    """ Les couleurs sont des nombres hexa """
    if re.match("^#[0-9A-F]{6}$", color, flags=re.I):
        return color
    raise ValidationError("color must be # + 6 hexadecimal digits (eg. #00ff00)")


def _hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def _rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(*rgb_color) # pylint: disable=consider-using-f-string

def _interpolate_color(color1, color2, factor):
    return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

class Color (models.Model):
    """ Dégradé de couleurs """
    name =  models.CharField(max_length=30, primary_key=True)
    steps = models.IntegerField(
            default=0,
            validators = [ _validate_steps ],
    )
    color = models.CharField(
            max_length=7,
            validators = [ _validate_color ],
    )
    color_max = models.CharField(
            max_length=7,
            default="#000000",
            validators = [ _validate_color ],
    )
    color_min = models.CharField(
            max_length=7,
            default="#FFFFFF",
            validators = [ _validate_color ],
    )

    def get_degraded(self):
        """ retourne le dégradé """
        colors = []
        start_rgb = _hex_to_rgb(self.color)
        white_rgb = _hex_to_rgb(self.color_min)
        black_rgb = _hex_to_rgb(self.color_max)
        half_steps = self.steps // 2
        # Generate lighter colors
        for i in range(half_steps):
            factor = (i + 1) / half_steps
            lighter_color = _interpolate_color(start_rgb, white_rgb, factor)
            colors.append(_rgb_to_hex(lighter_color))

        # Include the start color in the middle
        colors.append(self.color)

        # Generate darker colors
        for i in range(half_steps):
            factor = (i + 1) / half_steps
            darker_color = _interpolate_color(start_rgb, black_rgb, factor)
            colors.append(_rgb_to_hex(darker_color))

        colors.sort(reverse=True)
        return colors
    class Meta:
        """ Nom de l'app """
        app_label  = 'pluvieux'
