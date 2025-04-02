import ee
import geemap
import pandas as pd
from tqdm import tqdm

# Initialiser Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-baptistebarraque')

# Définir la zone d'intérêt
roi = ee.Geometry.Point([2.3522, 48.8566])  # Paris

# Définir la période
start_date = '2023-01-01'
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
 

def extract_data(date):
    date = ee.Date(date)
    
    temp = temperature \
        .filterDate(date, date.advance(1, 'day')) \
        .mean()
        
    precip = precipitation \
        .filterDate(date, date.advance(1, 'day')) \
        .mean()
        
    ndvi = vegetation \
        .filterDate(date, date.advance(1, 'month')) \
        .mean()
    
    values = ee.Feature(None, {
        'date': date.format('YYYY-MM-dd').getInfo(),
        'temperature': temp.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=1000
        ).get('soil_temperature_level_1').getInfo(),
        'precipitation': precip.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=1000
        ).get('total_precipitation_max').getInfo(),
        'ndvi': ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=1000
        ).get('NDVI').getInfo()
    })
    
    return values

# Créer la liste de dates
date_list = [ee.Date(start_date).advance(day, 'day') for day in range(350)]
data=[]
# Extraire les données
for date in tqdm(date_list, desc="Extracting data", unit="date"):
    data.append(extract_data(date))
# Convertir en DataFrame et sauvegarder
df = pd.DataFrame(data)
df.to_csv('environmental_data.csv', index=False)