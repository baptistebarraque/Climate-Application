
import ee
import geemap
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime as dt
import csv
import numpy as np



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

def get_daily_moist(start_date, end_date, point):
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)
    lon, lat = point
    point_geom = ee.Geometry.Point([lon, lat])

    humidity=collection_humidity.filterDate(start_date, end_date).select('SoilMoist_S_tavg')

    daily_moist = {}

    # Fonction pour calculer la précipitation moyenne pour une image
    def calculate_moist(image):
        date = image.date().format('YYYY-MM-dd').getInfo()
        moist_mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000
        ).get('SoilMoist_S_tavg').getInfo()
        return date, moist_mean

    # Utiliser ThreadPoolExecutor pour paralléliser les appels à getInfo()
    date_range = humidity.toList(humidity.size().getInfo())
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_moist, ee.Image(date_range.get(i)))
                   for i in range(date_range.size().getInfo())]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing days"):
            date, moist_mean = future.result()
            daily_moist[date] = moist_mean if moist_mean is not None else 0

    return daily_moist


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


def get_15_days_moist_average(point, end_date=None):
    """
    Calcule la moyenne des précipitations des 15 derniers jours pour un point donné.
    
    :param point: Tuple (longitude, latitude) du point d'intérêt.
    :param end_date: Date de fin (format 'YYYY-MM-DD'). Si None, utilise la date d'aujourd'hui.
    :return: Moyenne des précipitations des 15 derniers jours (en mm).
    """
    
    # Si pas de date de fin spécifiée, utiliser aujourd'hui
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        # S'assurer que end_date est au bon format
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Calculer la date de début (15 jours avant end_date)
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_obj = end_date_obj - timedelta(days=14)  # 14 jours avant pour avoir 15 jours au total
    start_date = start_date_obj.strftime('%Y-%m-%d')
    
    print(f"Récupération des humidités du {start_date} au {end_date}")
    print(f"Point: Longitude {point[0]}, Latitude {point[1]}")
    
    # Récupérer les données de précipitation
    daily_moist = get_daily_moist(start_date, end_date, point)
    
    # Calculer la moyenne
    if daily_moist:
        moist_values = list(daily_moist.values())
        average_moist = sum(moist_values) / len(moist_values)
        
        print(f"Nombre de jours traités: {len(moist_values)}")
        print(f"Précipitation moyenne sur 15 jours: {average_moist:.3f} mm")
        
        return {
            'average_precipitation': average_moist,
            'daily_data': daily_moist,
            'period': f"{start_date} to {end_date}",
            'num_days': len(moist_values)
        }
    else:
        print("Aucune donnée de précipitation trouvée")
        return {
            'average_precipitation': 0,
            'daily_data': {},
            'period': f"{start_date} to {end_date}",
            'num_days': 0
        }


def is_rainfall_event(date, point, precipitation_threshold=0.001):
    """
    Détermine si une date donnée correspond à un épisode pluvieux.

    :param date: Date à vérifier (format 'YYYY-MM-DD').
    :param point: Coordonnées (longitude, latitude) du point d'intérêt.
    :param precipitation_threshold: Seuil de précipitation pour définir un épisode pluvieux (en mm).
    :return: True si c'est un épisode pluvieux, False sinon.
    """
    date = ee.Date(date)
    lon, lat = point
    point_geom = ee.Geometry.Point([lon, lat])

    # Charger les données de précipitation pour la date spécifiée
    precip = collection_daily_aggr \
        .filterDate(date, date.advance(1, 'day')) \
        .select('total_precipitation_max') \
        .mean()

    # Calculer la précipitation moyenne pour le point donné
    precip_mean = precip.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000).get('total_precipitation_max').getInfo()

    # Vérifier si la précipitation dépasse le seuil
    if precip_mean is not None and precip_mean >= precipitation_threshold:
        return True
    return False

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

## Fonction qui récupère les données associées aux épisodes pluvieux
## Récupération des données effectuée en parallèle pour réduire le temps de chargement
## Cela peut prendre plusieurs heure pour des larges periodes de temps (plusieurs années)
def analyze_rainfall_episodes_parallel(start, rainfall_episodes, centroid):
    data_dict = {}
    i = 1
    
    with ThreadPoolExecutor() as executor:
        future_to_date = {}

        for date_str, episode in rainfall_episodes.items():
            index=start[:4]+'_'+str(i)
            start_date = datetime.strptime(date_str, '%Y-%m-%d')
            end_date = start_date + timedelta(days=episode['duration'] - 1)

            # Calculer les dates requises
            three_days_before_end = end_date - timedelta(days=3)
            one_day_before_end = end_date - timedelta(days=1)
            three_days_after_end = end_date + timedelta(days=3)

            # Soumettre les tâches pour extraire les données en parallèle
            future_to_date[executor.submit(extract_data, three_days_before_end.strftime('%Y-%m-%d'), centroid)] = ('three_days_before_end', index)
            future_to_date[executor.submit(extract_data, one_day_before_end.strftime('%Y-%m-%d'), centroid)] = ('one_day_before_end', index)
            future_to_date[executor.submit(extract_data, end_date.strftime('%Y-%m-%d'), centroid)] = ('end_date', index)
            future_to_date[executor.submit(extract_data, three_days_after_end.strftime('%Y-%m-%d'), centroid)] = ('three_days_after_end', index)

            # Ajouter les informations de base pour l'épisode
            data_dict[index] = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration': episode['duration'],
                'total_precipitation': episode['total_precipitation'],
                'longitude': centroid[0],
                'latitude': centroid[1],
                'temperature_3days_before_date': None,
                'temperature_1day_before_date': None,
                'temperature_end_date': None,
                'precipitation_3days_before_date': None,
                'precipitation_1day_before_date':None,
                'precipitation_end_date': None,
                'ndvi_3days_before_date': None,
                'ndvi_end_date': None,
                'lai_high_3days_before_date': None,
                'lai_high_1day_before_date':None,
                'lai_high_end_date': None,
                'lai_low_3days_before_date': None,
                'lai_low_1day_before_date':None,
                'lai_low_end_date': None,
                'albedo_3days_before_date': None,
                'albedo_1day_before_date':None,
                'albedo_end_date': None,
                'humidity_3days_before_date': None,
                'humidity_1day_before_date':None,
                'humidity_end_date': None,
                'humidity_3days_after_date': None,
            }
            i+=1

        # Récupérer les résultats des tâches avec une barre de progression
        for future in tqdm(as_completed(future_to_date), total=len(future_to_date), desc="Extracting data"):
            date_type, idx = future_to_date[future]
            try:
                data = future.result()
                if date_type == 'three_days_before_end':
                    data_dict[idx]['temperature_3days_before_date'] = data['temperature']
                    data_dict[idx]['precipitation_3days_before_date'] = data['precipitation']
                    data_dict[idx]['ndvi_3days_before_date'] = data['ndvi']
                    data_dict[idx]['lai_high_3days_before_date'] = data['lai_high']
                    data_dict[idx]['lai_low_3days_before_date'] = data['lai_low']
                    data_dict[idx]['albedo_3days_before_date'] = data['albedo']
                    data_dict[idx]['humidity_3days_before_date'] = data['humidity']
                elif date_type == 'end_date':
                    data_dict[idx]['temperature_end_date'] = data['temperature']
                    data_dict[idx]['precipitation_end_date'] = data['precipitation']
                    data_dict[idx]['ndvi_end_date'] = data['ndvi']
                    data_dict[idx]['lai_high_end_date'] = data['lai_high']
                    data_dict[idx]['lai_high_end_date'] = data['lai_high']
                    data_dict[idx]['albedo_end_date'] = data['albedo']
                    data_dict[idx]['humidity_end_date'] = data['humidity']
                elif date_type == 'one_day_before_end':
                    data_dict[idx]['temperature_1day_before_date']= data['temperature']
                    data_dict[idx]['precipitation_1day_before_date']=data['precipitation']
                    data_dict[idx]['ndvi_1day_before_date']=data['ndvi']
                    data_dict[idx]['lai_high_1day_before_date']=data['lai_high']
                    data_dict[idx]['lai_low_1day_before_date']=data['lai_low']
                    data_dict[idx]['albedo_1day_before_date']=data['albedo']
                    data_dict[idx]['humidity_1day_before_date']=data['humidity']
                elif date_type == 'three_days_after_end':
                    data_dict[idx]['humidity_3days_after_date'] = data['humidity']
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    return data_dict


## Fonction utilisée pour la prédiction (on enlève la valeur cible car on ne peut y avoir accès):
def analyze_rainfall_episodes_parallel_prediction(start, rainfall_episodes, centroid):
    data_dict = {}
    i = 1
    
    with ThreadPoolExecutor() as executor:
        future_to_date = {}

        for date_str, episode in rainfall_episodes.items():
            index=start[:4]+'_'+str(i)
            start_date = datetime.strptime(date_str, '%Y-%m-%d')
            end_date = start_date + timedelta(days=episode['duration'] - 1)

            # Calculer les dates requises
            three_days_before_end = end_date - timedelta(days=3)
            one_day_before_end = end_date - timedelta(days=1)

            # Soumettre les tâches pour extraire les données en parallèle
            future_to_date[executor.submit(extract_data, three_days_before_end.strftime('%Y-%m-%d'), centroid)] = ('three_days_before_end', index)
            future_to_date[executor.submit(extract_data, one_day_before_end.strftime('%Y-%m-%d'), centroid)] = ('one_day_before_end', index)
            future_to_date[executor.submit(extract_data, end_date.strftime('%Y-%m-%d'), centroid)] = ('end_date', index)

            # Ajouter les informations de base pour l'épisode
            data_dict[index] = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration': episode['duration'],
                'total_precipitation': episode['total_precipitation'],
                'longitude': centroid[0],
                'latitude': centroid[1],
                'temperature_3days_before_date': None,
                'temperature_1day_before_date': None,
                'temperature_end_date': None,
                'precipitation_3days_before_date': None,
                'precipitation_1day_before_date':None,
                'precipitation_end_date': None,
                'ndvi_3days_before_date': None,
                'ndvi_end_date': None,
                'lai_high_3days_before_date': None,
                'lai_high_1day_before_date':None,
                'lai_high_end_date': None,
                'lai_low_3days_before_date': None,
                'lai_low_1day_before_date':None,
                'lai_low_end_date': None,
                'albedo_3days_before_date': None,
                'albedo_1day_before_date':None,
                'albedo_end_date': None,
                'humidity_3days_before_date': None,
                'humidity_1day_before_date':None,
                'humidity_end_date': None,            }
            i+=1

        # Récupérer les résultats des tâches avec une barre de progression
        for future in tqdm(as_completed(future_to_date), total=len(future_to_date), desc="Extracting data"):
            date_type, idx = future_to_date[future]
            try:
                data = future.result()
                if date_type == 'three_days_before_end':
                    data_dict[idx]['temperature_3days_before_date'] = data['temperature']
                    data_dict[idx]['precipitation_3days_before_date'] = data['precipitation']
                    data_dict[idx]['ndvi_3days_before_date'] = data['ndvi']
                    data_dict[idx]['lai_high_3days_before_date'] = data['lai_high']
                    data_dict[idx]['lai_low_3days_before_date'] = data['lai_low']
                    data_dict[idx]['albedo_3days_before_date'] = data['albedo']
                    data_dict[idx]['humidity_3days_before_date'] = data['humidity']
                elif date_type == 'end_date':
                    data_dict[idx]['temperature_end_date'] = data['temperature']
                    data_dict[idx]['precipitation_end_date'] = data['precipitation']
                    data_dict[idx]['ndvi_end_date'] = data['ndvi']
                    data_dict[idx]['lai_high_end_date'] = data['lai_high']
                    data_dict[idx]['lai_high_end_date'] = data['lai_high']
                    data_dict[idx]['albedo_end_date'] = data['albedo']
                    data_dict[idx]['humidity_end_date'] = data['humidity']
                elif date_type == 'one_day_before_end':
                    data_dict[idx]['temperature_1day_before_date']= data['temperature']
                    data_dict[idx]['precipitation_1day_before_date']=data['precipitation']
                    data_dict[idx]['ndvi_1day_before_date']=data['ndvi']
                    data_dict[idx]['lai_high_1day_before_date']=data['lai_high']
                    data_dict[idx]['lai_low_1day_before_date']=data['lai_low']
                    data_dict[idx]['albedo_1day_before_date']=data['albedo']
                    data_dict[idx]['humidity_1day_before_date']=data['humidity']
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    return data_dict

## Fonction qui sépare la période totale en intervalles d'une année pour permettre à l'API de récupérer les données année par année.

from datetime import datetime, timedelta

def generate_yearly_intervals(start_date, end_date):
    """
    Génère des couples (start_date, end_date) avec un intervalle d'un an entre les dates.

    :param start_date: Date de début au format 'YYYY-MM-DD'.
    :param end_date: Date de fin au format 'YYYY-MM-DD'.
    :return: Liste de tuples (start_date, end_date) avec un intervalle d'un an.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    intervals = []
    
    while start < end:
        new_end = start + timedelta(days=365)  # Ajoute un an
        if new_end > end:
            new_end = end  # S'assure de ne pas dépasser la date finale
        
        intervals.append((start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d")))
        start = new_end  # Déplace la date de début au prochain intervalle
    
    return intervals

# Fonction qui récupère toutes les données nécessaire pour une période donnée sur une zone d'intérêt donnée
def get_data_over_period(start_period,end_period,centroid):
    years=generate_yearly_intervals(start_period, end_period)
    print(years)
    data={}
    current=1
    total=len(years)+1
    for start_date, end_date in years:
        print('Currently on year',start_date,end_date)
        daily_precipitations=get_daily_precipitation(start_date, end_date, centroid)
        rainfall_episodes=identify_rainfall_episodes(daily_precipitations)
        print(len(rainfall_episodes),' épisodes pluvieux cette année')
        yearly_data=analyze_rainfall_episodes_parallel(start_date,rainfall_episodes,centroid)
        data=data|yearly_data
        print(len(data))
    return data


## Génération des fichiers csv associés à la période et à la zone données récupérés à l'aide de la fonction précédente

def get_headers(data):
    headers = set()
    for entry in data.values():
        headers.update(entry.keys())
    return ['id'] + sorted(headers)
    
    # Écriture du fichier CSV
def export_to_csv(data, filename='output.csv'):
    headers = get_headers(data)
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        
        for key, values in data.items():
            row = {'id': key, **values}
            writer.writerow(row)
    print(f"Fichier CSV '{filename}' généré avec succès.")



## Main fonction qui prend en entrée un zone d'intérêt et qui renvoie l'excel associé avec toutes les données des différentes features pour chaque épisode pluvieux depuis 20 ans

def main_creation_csv(centroid, name_roi,filename):
    start_date='2010-01-01'             ### !!! Modifier cette date et remettre 2010 pour obtenir un trainingset suffisamment important.
    today = datetime.now()
    end_date = today - timedelta(days=30)
    end_date=end_date.strftime('%Y-%m-%d')
    data=get_data_over_period(start_date,end_date,centroid)
    export_to_csv(data,filename)
    df=pd.read_csv(filename)
    print(len(df),"épisodes pluvieux ont été télchargés")


## Fonction qui identifie le dernier épisode pluvieux en date et génère le csv pour la prédiction

def identify_last_rainfall(date,point):
    end_date = ee.Date(date)
    start_date = end_date.advance(-100, 'day')
    precipitation=get_daily_precipitation(start_date,end_date,point)
    rainfall_episodes=identify_rainfall_episodes(precipitation,precipitation_threshold=0.0005,secondary_threshold=0.00025)
    print(rainfall_episodes)
    last_three_items = list(rainfall_episodes.items())[-3:]
    last_episodes = dict(last_three_items)
    print(last_episodes)
    data=analyze_rainfall_episodes_parallel_prediction(date,last_episodes,point)
    export_to_csv(data,'predictions/prediction_'+date+'.csv')
    






## Données de test


coords = [
    [2.9006, 42.6875],
    [2.3539, 43.2163],
    [1.4428, 43.6048],
    [1.5218, 42.5063],
    [2.9006, 42.6875]  # Fermer le polygone
]
roi = ee.Geometry.Polygon(coords)







