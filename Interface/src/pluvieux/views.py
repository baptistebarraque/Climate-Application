""" main views for this project """
import random
from django.shortcuts import render
import ee
import folium
from folium import plugins  # pylint: disable=unused-import
from pluvieux.forms import MapForm
from pluvieux.models import Pmap
from datetime import datetime
from pluvieux.models import UserMapSettings
import json
from datetime import timedelta, datetime
def _add_layer(m, t, date):
    """ Ajourt une couche issue de la collection modis """
    h = t.get_params(date)
    
    end_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=30)
    
    print(date)
    start_date_str = start_date.strftime('%Y-%m-%d')
    print(start_date_str)
    _vis = {
        'min':     h['min'],
        'max':     h['max'],
        'palette': h['color'],
        'opacity': h['opacity']
    }
    
    myee = ee.ImageCollection(h['collection']).filter(
        ee.Filter.date(start_date_str, date))
    layer = myee.select(h['band'])
    folium.TileLayer(
        tiles=layer.getMapId(_vis)['tile_fetcher'].url_format,
        attr=h['band'],
        overlay=True,
        name=h['band'],
    ).add_to(m)



import ee
import folium
from datetime import timedelta, datetime
#Fonction de régression linéaire pour augmenter la granularité entre les différentes base de données.
def granulation_boost(date_str, coordinates, my_map):

    # Define region of interest and dates
    ROI = ee.Geometry.Rectangle(coordinates)  # Global example
    end_date = date_str
    date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = date-timedelta(days=5)
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Load Landsat images
    landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(ROI) \
        .filterDate(start_date, end_date)
    landsat_image = landsat_collection.median()  # Combine images

    # Load Sentinel-2 images
    sentinel_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(ROI) \
        .filterDate(start_date, end_date)
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
        ],
        'opacity': 0.7
    }

    # Create Folium map
    

    # Add Earth Engine layer
    folium.TileLayer(
        tiles=predicted_lst.getMapId(vis_params)['tile_fetcher'].url_format,
        attr='Map data &copy; <a href="https://www.google.com/earth">Google Earth</a>',
        name="Predicted LST"
    ).add_to(my_map)

    return my_map

#Fonction qui calcule les paramètres de la zone géométrique choisie
import geopy
from geopy.distance import distance

def get_square_bounds(center_lat, center_lon, distance_km):
    coordinates=[]
    # Calculer les coins du carré
    half_distance = distance_km / 2

    # Calculer les coordonnées des coins
    lat_min, lon_min = distance(kilometers=half_distance).destination((center_lat, center_lon), bearing=225).latitude, distance(kilometers=half_distance).destination((center_lat, center_lon), bearing=225).longitude
    lat_max, lon_max = distance(kilometers=half_distance).destination((center_lat, center_lon), bearing=45).latitude, distance(kilometers=half_distance).destination((center_lat, center_lon), bearing=45).longitude
    coordinates.append(lon_min)
    coordinates.append(lat_min)
    coordinates.append(lon_max)
    coordinates.append(lat_max)
    return coordinates
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

def cartes(request):
    """ Affiche la carte suite à une requête POST valide, sinon le formulaire """
    ee.Authenticate()
    ee.Initialize(project='ee-baptistebarraque')

    m = folium.Map(
        location=[46.529, 6.746],
        zoom_start=3,
        fadeAnimation=False,
        width="100%",
        tiles="Cartodb Positron"  #Carte blanche
    )

    # Par défaut, aucune carte et aucune date sélectionnées
    mymap = None
    selected_date = None
    distance=10000 # Taille du carré sur lequel seront effectué les calculs (km). Si le carré est trop petit, ou majoritairement dans la mer, des données seront manquantes et il ne sera pas possible de faire la régression

    if request.method == 'POST':
        form = MapForm(request.POST)
        if form.is_valid():
            # Récupère la carte et la date sélectionnées dans le formulaire
            mymap = form.cleaned_data['mymap']
            datestp = form.cleaned_data['date']
            adress = form.cleaned_data['adress']
            print(adress)
            date = datestp.strftime('%Y-%m-%d')

    else:
        # Sélectionne une carte aléatoire si aucune sélection via formulaire
        mymaps = list(Pmap.objects.all())
        mymap = random.choice(mymaps)
        # Objet datetime qui renvoie la date du jour si rien n'a été choisi
        datestp = datetime.today()
        datestp = datestp - timedelta(days=10)

        date = datestp.strftime('%Y-%m-%d')
        adress='Perpignan, France'
        
    
    folium.plugins.Geocoder().add_to(m)
   
    coordinates=get_coordinates(adress)
    center_lat, center_lon = coordinates
    #Calcule les coordoonées de la zone d'intérêt où se concentreront les calculs
    zone_coordinates = get_square_bounds(center_lat, center_lon, distance)
    rounded_coordinates = [round(num, 1) for num in zone_coordinates]   #Nécessité d'arrondir sinon l'algorithme ne fonctionne pas
    print(zone_coordinates)
    print(rounded_coordinates)
    print(date)
    # Ajout du layer de température augmentée
    m=granulation_boost(date,rounded_coordinates,m)
    # Ajoute les couches de la carte sélectionnée
    _add_layer(m, mymap.layer1, date)
    _add_layer(m, mymap.layer2, date)
    _add_layer(m, mymap.layer3, date)
    
    # Ajoute des plugins à la carte Folium
    folium.plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)
    if 'tile_url' in request.session:
        folium.TileLayer(
            tiles=request.session['tile_url'],
            attr='Landsat Image',
            name='Landsat Image',
            overlay=True
        ).add_to(m)

    
    folium.plugins.LocateControl(initialZoomLevel=3).add_to(m)

    # Génère le rendu HTML de la carte
    figure = folium.Figure()
    m.add_to(figure)
    figure.render()

    # Recharge le formulaire pour la prochaine requête
    form = MapForm()

    # Liste des paramètres des couches (juste pour affichage)
    parameter_list = [
        'Predicted LST',
        mymap.layer1.description,
        mymap.layer2.description,
        mymap.layer3.description,
        
    ]

    # Rendu final
    return render(request, 'map_render.html', {
        'map': figure,
        'map_name': m.get_name(),
        'parameter_list': parameter_list,
        'form': form,
        'selected_date': selected_date,  # Passe la date sélectionnée au template
    })


from django.http import JsonResponse


def load_map_settings(request):
    if request.method == "GET" and request.user.is_authenticated:
        name = request.GET.get("name")
        if not name:
            return JsonResponse({"error": "Name parameter is required."}, status=400)
        try:
            user_settings = UserMapSettings.objects.get(user=request.user, name=name)
            print(f"Settings found: {user_settings.location}, {user_settings.zoom}, {user_settings.date}")  # Debug print
            return JsonResponse({
                "location": user_settings.location,
                "zoom": user_settings.zoom,
                
            }, status=200)
        except UserMapSettings.DoesNotExist:
            return JsonResponse({"error": "No settings found."}, status=404)
    return JsonResponse({"error": "Unauthorized."}, status=403)

def save_map_settings(request):
    if request.method == "POST" and request.user.is_authenticated:
        try:
            data = json.loads(request.body)
            name = data.get("name")
            location = data.get("location")
            zoom = data.get("zoom")
            

            if not name:
                return JsonResponse({"status": "error", "message": "Name parameter is required."}, status=400)

            user_settings, created = UserMapSettings.objects.get_or_create(user=request.user, name=name)
            user_settings.location = location
            user_settings.zoom = zoom
            
            user_settings.save()

            return JsonResponse({"status": "success", "message": "Paramètres sauvegardés."}, status=200)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error", "message": "Requête non autorisée."}, status=403)

def get_all_map_settings(request):
    if request.method == "GET" and request.user.is_authenticated:
        user_settings = UserMapSettings.objects.filter(user=request.user)
        settings_list = [
            {
                "name": setting.name,
                "location": setting.location,
                "zoom": setting.zoom,
                
            }
            for setting in user_settings
        ]
        return JsonResponse({"settings": settings_list}, status=200)
    return JsonResponse({"error": "Unauthorized."}, status=403)


def delete_map_settings(request):
    if request.method == "DELETE" and request.user.is_authenticated:
        try:
            data = json.loads(request.body)
            name = data.get("name")

            if not name:
                return JsonResponse({"status": "error", "message": "Name parameter is required."}, status=400)

            try:
                user_settings = UserMapSettings.objects.get(user=request.user, name=name)
                user_settings.delete()
                return JsonResponse({"status": "success", "message": "Paramètres supprimés avec succès."}, status=200)
            except UserMapSettings.DoesNotExist:
                return JsonResponse({"status": "error", "message": "Paramètres introuvables."}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error", "message": "Requête non autorisée."}, status=403)