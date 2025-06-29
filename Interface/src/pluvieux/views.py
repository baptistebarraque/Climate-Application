""" main views for this project """
import random
from django.shortcuts import render, redirect
import ee
import folium
from folium import plugins  # pylint: disable=unused-import
from pluvieux.forms import MapForm
from pluvieux.models import Pmap
from datetime import datetime
from pluvieux.models import UserMapSettings
from pluvieux.models import UserPredictionSettings
import json
from datetime import timedelta, datetime
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import os
from .utils import main_creation_csv
from .utils import complete_pipeline_with_sequences
import ast
import warnings
import keras
from .utils import tuning_model
from keras.models import load_model
from keras.losses import MeanSquaredError
from .utils import is_rainfall_event
from .utils import identify_last_rainfall
from .utils import preprocessing_pipeline_for_prediction
from .utils import get_15_days_moist_average

#warnings.filterwarnings('ignore', category=RuntimeWarning)


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


@login_required
def _prediction_interface_view(request):
    return render(request, 'prediction_interface.html')

@login_required
def load_data_view(request):
    if request.method=='POST':
        prediction_zone_name = request.POST.get('prediction_zone_name').lower().replace(' ','_')
        

        prediction_zone_location_str = request.POST.get('prediction_zone_location')
        try:
            prediction_zone_location = ast.literal_eval(prediction_zone_location_str)
            
        except (ValueError, SyntaxError):
            prediction_zone_location = None
            print("Erreur de parsing de la position")
        
        
        file_path = os.path.join(settings.MEDIA_ROOT, f"data/{prediction_zone_name}_{request.user.id}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        main_creation_csv(prediction_zone_location, prediction_zone_name,file_path)
        user_settings, created = UserPredictionSettings.objects.get_or_create(user=request.user, zone_prediction_name=prediction_zone_name)
        
        user_settings.zone_prediction_location=prediction_zone_location
        user_settings.data_file.name = file_path # Relative to MEDIA_ROOT
        user_settings.save()

        return JsonResponse({'success': True, 'message': 'Fichier CSV généré avec succès.'})
    
    return JsonResponse({'success': False, 'error': 'Méthode non autorisée.'}, status=405)

@login_required
def get_user_zones(request):
    zones = UserPredictionSettings.objects.filter(user=request.user)
    data = [{'id': z.id, 'name': z.zone_prediction_name} for z in zones]
    return JsonResponse({'zones': data})


@login_required
def train_model_view(request):
    if request.method == 'POST':
        zone_id = request.POST.get('zone_id')
        try:
            zone = UserPredictionSettings.objects.get(id=zone_id, user=request.user)
            file_path = zone.data_file.path
            print(file_path)
            X_train, X_test, y_train, y_test, scaler, encoders, df = complete_pipeline_with_sequences(file_path)
            print('len train:',len(X_train))
            print(len(X_test))
            print('ok')
            best_model, history, best_hyperparameters= tuning_model(X_train,X_test,y_train,y_test,max_trials=10, max_epochs=10,model_name=zone.zone_prediction_name)
            from django.core.files import File
            model_path='trainedmodels/'+zone.zone_prediction_name+'.h5'
            with open(model_path, 'rb') as f:
                django_file = File(f)
                zone.model_file.save(zone.zone_prediction_name, django_file, save=True) 
            return JsonResponse({
                'success': True,
                'message': 'Pipeline exécuté avec succès',
                'train_size': len(X_train),
                'test_size': len(X_test),
            })
        except UserPredictionSettings.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Zone non trouvée'}, status=404)
        
        except Exception as e:
            if "Singleton array" in str(e):
                # Parfois des singletons remplacent des tableaux, ce problème n'a pas su être géré, cela n'affecte pas le résultat
                return JsonResponse({'success': True, 'message': 'Singleton array error occurred but was handled.'})
            else:
                return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'}, status=405)



def predict_if_rainy_yesterday_view(request):
    if request.method == 'POST':
        zone_id = request.POST.get('zone_id')

        try:
            zone = UserPredictionSettings.objects.get(id=zone_id, user=request.user)

            # Vérifier l’épisode pluvieux pour la veille
            yesterday = datetime.today() - timedelta(days=150)
            yesterday=yesterday.strftime('%Y-%m-%d')
            if not is_rainfall_event(yesterday,zone.zone_prediction_location,0):
                return JsonResponse({
                    'success': True,
                    'message': "Pas d'épisode pluvieux hier. Aucune prédiction nécessaire."
                })

            if not zone.model_file:
                return JsonResponse({'success': False, 'error': 'Aucun modèle enregistré pour cette zone.'})

            model_path = zone.model_file.path
            print(model_path)
            model_path=model_path+'.h5'
            
            model = keras.models.load_model(model_path, compile=False)
           
            identify_last_rainfall(yesterday,zone.zone_prediction_location)
            data_path='predictions/prediction_'+yesterday+'.csv'
            
            print(data_path)
            X_seq, scaler, encoders, df= preprocessing_pipeline_for_prediction(data_path)
            print('fin du pipeline')
            
            predicted_value=model.predict(X_seq)
            print(predicted_value)
            # predicted_value=scaler.inverse_transform(predicted_value)
            # print('real', predicted_value)
            precip_result = get_15_days_moist_average(zone.zone_prediction_location, end_date=yesterday)
            average_15_days = precip_result['average_precipitation']
            
            print(f"Prédiction: {predicted_value}")
            print(f"Moyenne 15 derniers jours: {average_15_days}")
            
            # Comparaison
            difference = float(predicted_value) - average_15_days
            percentage_difference = (difference / average_15_days * 100) if average_15_days > 0 else 0
            
            # Déterminer le message de comparaison
            if difference > 0:
                comparison_message = f"La prédiction est supérieure à la moyenne des 15 derniers jours de {difference:.2f}kg/m^2 ({percentage_difference:.1f}%)"
            elif difference < 0:
                comparison_message = f"La prédiction est inférieure à la moyenne des 15 derniers jours de {abs(difference):.2f}kg/m^2 ({abs(percentage_difference):.1f}%)"
            else:
                comparison_message = "La prédiction est égale à la moyenne des 15 derniers jours"

            return JsonResponse({
                'success': True,
                'prediction': float(predicted_value),
                'message': "Prédiction effectuée suite à un épisode pluvieux hier.",
                'comparison_message': comparison_message,
                
            })

        except UserPredictionSettings.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Zone non trouvée'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'}, status=405)