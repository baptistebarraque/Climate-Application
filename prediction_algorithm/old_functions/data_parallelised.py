import ee
import geemap
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Initialiser Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-baptistebarraque')

# Définir la zone d'intérêt (rectangle)
# Coordonnées des coins du rectangle (longitude, latitude) 
#Zone entre Perpignan, Carcassonne, Toulouse et Andore la ville
coords = [
    [2.9006, 42.6875],
    [2.3539, 43.2163],
    [1.4428, 43.6048],
    [1.5218, 42.5063],
    [2.9006, 42.6875]  # Fermer le polygone
]
roi = ee.Geometry.Polygon(coords)

# Définir la période
start_date = '2013-01-01'
end_date = '2023-12-31'

# Charger les collections
temperature = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
    .filterDate(start_date, end_date) \
    .select('soil_temperature_level_1')

precipitation = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
    .filterDate(start_date, end_date) \
    .select('total_precipitation_max')

vegetation = ee.ImageCollection('MODIS/061/MOD13A1') \
    .filterDate(start_date, end_date) \
    .select('NDVI')

# Fonction pour créer une grille de points à l'intérieur du rectangle
def create_grid(roi, scale):
    bounds = roi.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]

    lon_range = range(int(min_lon * 1000), int(max_lon * 1000), int(scale * 1000))
    lat_range = range(int(min_lat * 1000), int(max_lat * 1000), int(scale * 1000))

    grid_points = []
    for lon in lon_range:
        for lat in lat_range:
            grid_points.append((lon / 1000.0, lat / 1000.0))

    return grid_points

# Créer la grille de points
grid_points = create_grid(roi, scale=0.01)  # Résolution de 0.01 degré

def extract_data(date, point):
    date = ee.Date(date)
    lon, lat = point
    point_geom = ee.Geometry.Point([lon, lat])

    temp = temperature \
        .filterDate(date, date.advance(1, 'day')) \
        .mean()

    precip = precipitation \
        .filterDate(date, date.advance(1, 'day')) \
        .mean()

    ndvi = vegetation \
        .filterDate(date, date.advance(1, 'month')) \
        .mean()

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
        ).get('NDVI').getInfo()
    }

    return values






# Créer la liste de dates avec des tranches de 5 jours tous les 10 jours
date_list = []
current_date = ee.Date(start_date)
total_days = (ee.Date(end_date).millis().getInfo() - ee.Date(start_date).millis().getInfo()) // (1000 * 60 * 60 * 24)

with tqdm(total=total_days, desc="Creating date list") as pbar:
    while current_date.millis().getInfo() <= ee.Date(end_date).millis().getInfo():
        for i in range(5):
            date_list.append(current_date.advance(i, 'day'))
        current_date = current_date.advance(10, 'day')
        pbar.update(10)
if current_date.millis().getInfo() <= ee.Date(end_date).millis().getInfo():
    for i in range(min(5, (ee.Date(end_date).millis().getInfo() - current_date.millis().getInfo()) // (1000 * 60 * 60 * 24) + 1)):
        date_list.append(current_date.advance(i, 'day'))
    pbar.update(min(5, (ee.Date(end_date).millis().getInfo() - current_date.millis().getInfo()) // (1000 * 60 * 60 * 24) + 1))
print(len(date_list))
data = []
retry_tasks = []
# Utiliser ThreadPoolExecutor pour paralléliser l'extraction des données
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_task = {executor.submit(extract_data, date, point): (date, point) for date in date_list for point in grid_points}
    
    for future in tqdm(as_completed(future_to_task), total=len(date_list) * len(grid_points), desc="Extracting data", unit="point"):
        date, point = future_to_task[future]
        try:
            data.append(future.result())
        except Exception as exc:
            print(f"Task for {date}, {point} generated an exception: {exc}")
            retry_tasks.append((date, point))

# Retry failed tasks
for attempt in range(100):
    if not retry_tasks:
        break  # Exit if there are no more tasks to retry

    print(f"Retry attempt {attempt + 1}")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {executor.submit(extract_data, date, point): (date, point) for date, point in retry_tasks}
        retry_tasks = []

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Retrying failed tasks", unit="point"):
            date, point = future_to_task[future]
            try:
                data.append(future.result())
            except Exception as exc:
                print(f"Retry task for {date}, {point} failed with exception: {exc}")
                retry_tasks.append((date, point))

# Convertir en DataFrame et sauvegarder
df = pd.DataFrame(data)
df.to_csv('environmental_data.csv', index=False)


df['date'] = pd.to_datetime(df['date'])

# Fonction pour calculer la moyenne des températures sur le mois précédent
def calculate_monthly_average(df, date_col, temp_col):
    df = df.sort_values(by=date_col)  # Assurez-vous que les données sont triées par date
    monthly_averages = []

    for index, row in df.iterrows():
        current_date = row[date_col]
        start_date = current_date - pd.DateOffset(months=1)
        mask = (df[date_col] >= start_date) & (df[date_col] < current_date)
        monthly_avg = df.loc[mask, temp_col].mean()
        monthly_averages.append(monthly_avg)

    return monthly_averages

