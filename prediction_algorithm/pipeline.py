### Implémentation du Pipeline global ###

import ee
import geemap
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Initialiser Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-baptistebarraque')

# Chargement des collections: 

collection_daily_aggr=ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
collection_modis=ee.ImageCollection('MODIS/061/MOD13A1')
collection_humidity=ee.ImageCollection('NASA/GLDAS/V022/CLSM/G025/DA1D')

### Fonction de récupération des données (prend en entrée une date et une localisation)
### Renvoie un dictionnaire avec les valeurs de chaque donnée pour la date recherchée

def extract_data(date, point):
    date = ee.Date(date)
    lon, lat = point
    point_geom = ee.Geometry.Point([lon, lat])

    # Chargement de la bande pour la température
    temp = collection_daily_aggr \
        .filterDate(date, date.advance(1, 'day')) \
        .select('soil_temperature_level_1') \
        .mean()

    # Chargement de la bande pour la precipitation
    precip = collection_daily_aggr \
        .filterDate(date, date.advance(1, 'day')) \
        .select('total_precipitation_max') \
        .mean()

    # Chargement de la bande pour l'indice NDVI
    ndvi = collection_modis \
        .filterDate(date, date.advance(1, 'month')) \
        .select('NDVI') \
        .mean()

    # Chargement de la bande pour l'indice LAI (high correspônd à la végétation haute (arbres) et low correspond à la végétation basse (herbe/champs))
    lai_high = collection_daily_aggr.filterDate(date, date.advance(1, 'week')) \
        .select('leaf_area_index_high_vegetation_max') \
        .mean()

    lai_low= collection_daily_aggr.filterDate(date, date.advance(1, 'week')) \
        .select('leaf_area_index_low_vegetation_max') \
        .mean()

    # Chargement de la bande pour l'albedo
    albedo = collection_daily_aggr.filterDate(date, date.advance(1, 'week')) \
            .select('forecast_albedo_max') \
            .mean()
    
    # Chargement de la bande pour l'humidité
    humidity=collection_humidity.filterDate(date, date.advance(1, 'week')).select('SoilMoist_S_tavg').mean()
    

    values = {
        'date': date.format('YYYY-MM-dd').getInfo(),
        'longitude': lon,
        'latitude': lat,
        
        'temperature': temp.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000
        ).get('soil_temperature_level_1').getInfo(),
        'precipitation': precip.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000
        ).get('total_precipitation_max').getInfo(),
        'ndvi': ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000
        ).get('NDVI').getInfo(),
        'lai_high': lai_high.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=5000
        ).get('leaf_area_index_high_vegetation_max').getInfo(),
        'lai_low': lai_low.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=5000
        ).get('leaf_area_index_low_vegetation_max').getInfo(),
        'albedo': albedo.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=5000
        ).get('forecast_albedo_max').getInfo(),
        'humidity':humidity.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=5000
        ).get('SoilMoist_S_tavg').getInfo(),

    }

    return values


########## Détection des épisodes pluvieux ##########
## Fonction qui renvoie la précipitation journalière pour une période temporelle donnée
# Renvoie un dictionnaire {date: précipitation}

def get_daily_precipitation(start_date, end_date, point):
    """
    Renvoie un dictionnaire avec les dates comme clés et les niveaux de précipitation comme valeurs pour une région d'intérêt.

    :param start_date: Date de début (format 'YYYY-MM-DD').
    :param end_date: Date de fin (format 'YYYY-MM-DD').
    :param roi: Géométrie de la région d'intérêt (ee.Geometry).
    :return: Dictionnaire avec les dates comme clés et les niveaux de précipitation comme valeurs.
    """
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)
    lon, lat = point
    point_geom = ee.Geometry.Point([lon, lat])

    # Charger les données de précipitation pour la période spécifiée
    precipitation_collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_date, end_date) \
        .select('total_precipitation_max')

    # Dictionnaire pour stocker les dates et les niveaux de précipitation
    daily_precipitation = {}

    # Fonction pour calculer la précipitation moyenne pour une image
    def calculate_precipitation(image):
        date = image.date().format('YYYY-MM-dd').getInfo()
        precip_mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000
        ).get('total_precipitation_max').getInfo()
        return date, precip_mean

    # Utiliser ThreadPoolExecutor pour paralléliser les appels à getInfo()
    date_range = precipitation_collection.toList(precipitation_collection.size().getInfo())
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_precipitation, ee.Image(date_range.get(i)))
                   for i in range(date_range.size().getInfo())]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing days"):
            date, precip_mean = future.result()
            daily_precipitation[date] = precip_mean if precip_mean is not None else 0

    return daily_precipitation


## Fonction qui identifie les épisodes pluvieux, leur durée et le total des précipitations

def identify_rainfall_episodes(daily_precipitation, precipitation_threshold=1e-3, secondary_threshold=5e-4):
    """
    Identifie les épisodes pluvieux, leur durée et le total des précipitations à partir d'un dictionnaire de précipitations quotidiennes.

    :param daily_precipitation: Dictionnaire avec les dates comme clés et les niveaux de précipitation comme valeurs.
    :param precipitation_threshold: Seuil de précipitation pour définir le début d'un épisode pluvieux
    :param secondary_threshold: Seuil secondaire pour définir la fin d'un épisode pluvieux
    :return: Dictionnaire avec les dates de début des épisodes pluvieux, leur durée et le total des précipitations.
    """
    # Liste pour stocker les dates de début des épisodes pluvieux, leur durée et le total des précipitations
    rainfall_episodes = {}
    current_episode_start = None
    current_episode_duration = 0
    current_episode_total_precip = 0

    # Analyser les résultats pour identifier les épisodes pluvieux
    for date, precip_mean in sorted(daily_precipitation.items()):
        # Vérifier si la précipitation dépasse le seuil principal
        if precip_mean is not None and precip_mean >= precipitation_threshold:
            if current_episode_start is None:
                current_episode_start = date
            current_episode_duration += 1
            current_episode_total_precip += precip_mean
        # Vérifier si la précipitation est entre le seuil principal et le seuil secondaire
        elif precip_mean is not None and precip_mean >= secondary_threshold:
            if current_episode_start is not None:
                current_episode_duration += 1
                current_episode_total_precip += precip_mean
        else:
            if current_episode_start is not None:
                rainfall_episodes[current_episode_start] = {
                    'duration': current_episode_duration,
                    'total_precipitation': current_episode_total_precip
                }
                current_episode_start = None
                current_episode_duration = 0
                current_episode_total_precip = 0

    # Ajouter le dernier épisode s'il est encore en cours
    if current_episode_start is not None:
        rainfall_episodes[current_episode_start] = {
            'duration': current_episode_duration,
            'total_precipitation': current_episode_total_precip
        }

    return rainfall_episodes

