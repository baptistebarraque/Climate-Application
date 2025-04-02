### Fonctions non utilisées au final ###

# Fonctions qui récupère les données sur 8 points autour du la zone d'intérêt


def extract_data_for_date_point(date, point, localisation_point):
    return extract_data(date, point, localisation_point)

def analyze_rainfall_episodes_parallel(points, rainfall_episodes):
    results = {}
    index=0
    for episode_date, episode_info in rainfall_episodes.items():
        
        start_date = datetime.strptime(episode_date, '%Y-%m-%d')
        duration = episode_info['duration']
        intensity = episode_info['total_precipitation']

        episode_results = {
            'index':index,
            'duration': duration,
            'intensity': intensity,
            'pre_episode_data': {},
            'episode_data': {}
        }

        all_tasks = []

        # Collect tasks for pre-episode data
        for days_before in range(1, 4):
            date_before = start_date - timedelta(days=days_before)
            date_str = date_before.strftime('%Y-%m-%d')
            episode_results['pre_episode_data'][date_str] = []

            for localisation_point in points.keys():
                all_tasks.append((date_str, points[localisation_point],localisation_point, 'pre_episode_data', date_str))

        # Collect tasks for episode data
        for day in range(duration):
            episode_day = start_date + timedelta(days=day)
            date_str = episode_day.strftime('%Y-%m-%d')
            episode_results['episode_data'][date_str] = []

            for localisation_point in points.keys():
                all_tasks.append((date_str, points[localisation_point],localisation_point, 'episode_data', date_str))

        # Execute tasks in parallel
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_data_for_date_point, date, point, localisation_point): (date, point,localisation_point, data_type, key)
                       for date, point,localisation_point, data_type, key in all_tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {episode_date}"):
                date, point,localisation_point, data_type, key = futures[future]
                try:
                    data = future.result()
                    episode_results[data_type][key].append(data)
                except Exception as e:
                    print(f"Error processing {date} at {point}: {e}")

        results[episode_date] = episode_results
        index+=1

    return results



### Fonction qui génère huit points autour du centroid de la zone d'intérêt:

#renvoie une liste qui comporte aussi le centroîde de la zone d'intérêt

import math
import ee


def points_autour_polygone(roi, distance, nb_points):
    # Calculer le centre du polygone
    centroid = roi.centroid().coordinates()
    center_lon, center_lat = centroid.getInfo()[0], centroid.getInfo()[1]
    centroid_coordinates = (center_lon, center_lat)

    # Fonction pour convertir les degrés en radians
    def deg_to_rad(deg):
        return deg * (math.pi / 180)

    # Fonction pour convertir les radians en degrés
    def rad_to_deg(rad):
        return rad * (180 / math.pi)

    # Dictionnaire pour stocker les coordonnées des points
    points = {"centre": centroid_coordinates}

    # Calculer les coordonnées des points autour du centre
    for i in range(nb_points):
        angle = deg_to_rad(45 * i)
        dx = distance * math.cos(angle)
        dy = distance * math.sin(angle)

        # Convertir les déplacements en coordonnées géographiques
        new_lat = center_lat + (dy / 111111)  # Approximation pour 1 degré de latitude ~ 111 km
        new_lon = center_lon + (dx / (111111 * math.cos(deg_to_rad(center_lat))))  # Approximation pour 1 degré de longitude

        points[i] = (new_lon, new_lat)

    return points



### Fonction qui calcule le nombre de points d'une grille donnée: 

def calculate_grid_points(coords, scale):
    """
    Calcule le nombre de points de grille pour une région d'intérêt donnée et une échelle spécifiée.

    :param coords: Liste de coordonnées définissant le polygone de la région d'intérêt.
    :param scale: Échelle de la grille en degrés.
    :return: Nombre total de points de grille.
    """
    # Extraire les limites de la région d'intérêt
    min_lon = min(coord[0] for coord in coords)
    max_lon = max(coord[0] for coord in coords)
    min_lat = min(coord[1] for coord in coords)
    max_lat = max(coord[1] for coord in coords)

    # Calculer le nombre de points dans chaque dimension
    num_lon_points = int((max_lon - min_lon) / scale) + 1
    num_lat_points = int((max_lat - min_lat) / scale) + 1

    # Calculer le nombre total de points
    total_points = num_lon_points * num_lat_points

    return total_points

### Fonction qui crée une grille avec un certain pas:

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

