import ee
import folium
from datetime import timedelta, datetime


my_map = folium.Map(
        location=[46.529, 6.746],
        zoom_start=3,
        fadeAnimation=False,
        width="100%",
        tiles="Cartodb Positron"  #Carte blanche
    )
def granulation_boost(date_str, coordinates, my_map):
# Authentication and initialization of Earth Engine
    ee.Authenticate()
    ee.Initialize(project='ee-baptistebarraque')

    # Define region of interest and dates
    ROI = ee.Geometry.Rectangle(coordinates)  # Global example
    end_date = date_str
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = date-timedelta(days=5)
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Load Landsat images
    landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(ROI) \
        .filterDate(start_date_str, date)
    landsat_image = landsat_collection.median()  # Combine images

    # Load Sentinel-2 images
    sentinel_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(ROI) \
        .filterDate(start_date_str, date)
    sentinel_image = sentinel_collection.median()

    # Calculate indices
    ndvi = sentinel_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = sentinel_image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = sentinel_image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    # Convert LST to Celsius
    lst = landsat_image.select('ST_B10') \
        .multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    # Create a single image with all predictors
    predictors = ndvi.addBands(ndwi).addBands(ndbi)

    # Create training points
    training_points = ee.FeatureCollection.randomPoints(ROI, 1000)

    # Sample training data
    training_data = predictors.addBands(lst).sampleRegions(
        collection=training_points,
        scale=30,
        geometries=True
    )

    # Perform linear regression using Earth Engine's built-in reducer
    regression = training_data.reduceColumns(
        reducer=ee.Reducer.linearRegression(
            numX=3,  # number of predictor variables
            numY=1   # number of dependent variables
        ),
        selectors=['NDVI', 'NDWI', 'NDBI', 'LST']
    )

    # Extract coefficients
    coefficients = ee.Array(regression.get('coefficients'))

    # Function to apply linear model
    def apply_linear_model(image):
        # Multiply each predictor by its coefficient and sum
        ndvi_pred = image.select('NDVI').multiply(coefficients.get([0,0]))
        ndwi_pred = image.select('NDWI').multiply(coefficients.get([1,0]))
        ndbi_pred = image.select('NDBI').multiply(coefficients.get([2,0]))
        
        # Sum the predictions
        return ndvi_pred.add(ndwi_pred).add(ndbi_pred)

    # Apply the linear model to predict LST
    predicted_lst = apply_linear_model(predictors)

    # Visualization parameters
    vis_params = {
        'min': 0,  # Minimum temperature in your dataset
        'max': 50,  # Maximum temperature in your dataset
        'palette': [
            '#0000FF',   # Deep Blue (coldest)
            '#00FFFF',   # Cyan
            '#00FF00',   # Bright Green
            '#FFFF00',   # Yellow
            '#FFA500',   # Orange
            '#FF0000'    # Deep Red (hottest)
        ]
    }

    # Create Folium map
    

    # Add Earth Engine layer
    folium.TileLayer(
        tiles=predicted_lst.getMapId(vis_params)['tile_fetcher'].url_format,
        attr='Map data &copy; <a href="https://www.google.com/earth">Google Earth</a>',
        name="Predicted LST"
    ).add_to(my_map)

    # Save the map
    my_map.save('lst_map.html')
    
    # Print coefficients for verification
    print("Regression Coefficients:")
    print(coefficients.getInfo())
    return my_map
granulation_boost('2024-11-30',[-5.0, 41.0, 9.5, 51.5], my_map)

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def get_coordinates(address):
    try:
        # Création du géocodeur avec un user_agent spécifique (obligatoire)
        geolocator = Nominatim(user_agent="my_application")
        
        # Obtention des coordonnées
        location = geolocator.geocode(address)
        
        if location:
            return location.latitude, location.longitude
        else:
            return None
            
    except GeocoderTimedOut:
        return None

# Exemple d'utilisation
adresse = "Saint Lattier"
coordinates = get_coordinates(adresse)
print(coordinates)