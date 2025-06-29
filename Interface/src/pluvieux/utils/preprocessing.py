import re
import pandas as pd
from datetime import datetime, timedelta
import datetime as dt
import csv
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)





##Fonctions de preprocessing qui permettent de rajouter des features et de préparer les données à l'entraînement du modèle


def load_data(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path,skiprows=[1,2])
    elif path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        raise ValueError("Format de fichier non supporté. Utilisez CSV ou Excel.")
    df.columns = [re.sub(r'[\*]', '', col) for col in df.columns]
    
    # Convertir les colonnes de dates en format datetime
    date_columns = ['end_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

## Fonction qui génère et ajoute au Dataframe les features 'mois' et 'saison':

def add_month_season(df, colonne_date='end_date'):
    # Convertir la colonne de dates en format datetime si ce n'est pas déjà fait
    df[colonne_date] = pd.to_datetime(df[colonne_date], errors='coerce')
    
    # Ajouter une colonne pour le mois
    df['Month'] = df[colonne_date].dt.strftime('%B')
    
    # Ajouter une colonne pour la saison
    def obtenir_saison(mois):
        if mois in [12, 1, 2]:
            return 'Winter'
        elif mois in [3, 4, 5]:
            return 'Spring'
        elif mois in [6, 7, 8]:
            return 'Summer'
        elif mois in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Unknown'
    
    df['Season'] = df[colonne_date].dt.month.apply(obtenir_saison)
    
    # Supprimer les colonnes de dates
    columns_to_drop = []
    if 'end_date' in df.columns:
        columns_to_drop.append('end_date')
    if 'start_date' in df.columns:
        columns_to_drop.append('start_date')
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    return df


##Fonction qui ajoute des features derivées au dataframe
def add_derivate_features(df):
    # Calculer la différence entre humidity_end_date et humidity_1day_before_date
    df['humidity_diff_end_1day'] = df['humidity_end_date'] - df['humidity_1day_before_date']

    # Calculer le taux de changement
    df['humidity_rate_of_change'] = (df['humidity_end_date'] - df['humidity_1day_before_date']) / (df['humidity_1day_before_date'] - df['humidity_3days_before_date'])

    # Calculer le rapport precipitation_end_date / humidity_end_date
    df['precipitation_humidity_ratio'] = df['precipitation_end_date'] / df['humidity_end_date']

    # Calculer le produit temperature_end_date * humidity_end_date
    df['temperature_humidity_product'] = df['temperature_end_date'] * df['humidity_end_date']

    # Calculer le rapport total_precipitation / duration
    df['total_precipitation_duration_ratio'] = df['total_precipitation'] / df['duration']

    return df

## Fonction qui gère les valeurs manquantes en utilisant la stratégie des k plus proches voisins:

# On peut questionner cette fonction


def generate_missing_values(df, target_column='humidity_3days_after_date'):
    """
    Gère les valeurs manquantes en utilisant différentes stratégies selon les colonnes

    Args:
        df: DataFrame pandas avec valeurs manquantes
        target_column: Nom de la colonne cible

    Returns:
        DataFrame avec valeurs manquantes traitées
    """
    df_copie = df.copy()

    # Supprimer les lignes où la variable cible est manquante
    df_copie = df_copie.dropna(subset=[target_column])

    # Identifier les colonnes numériques après suppression
    numeric_columns = df_copie.select_dtypes(include=[np.number]).columns.tolist()

    # Supprimer les colonnes entièrement manquantes
    numeric_columns = [col for col in numeric_columns if not df_copie[col].isnull().all()]

    # Pour les autres colonnes numériques, utiliser KNN Imputer
    if len(numeric_columns) > 0:
        # Créer une copie des colonnes numériques pour l'imputation
        df_numeric = df_copie[numeric_columns].copy()

        # Initialiser l'imputeur KNN et l'appliquer
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_numeric),
            columns=numeric_columns,
            index=df_copie.index
        )

        # Remplacer les valeurs dans le DataFrame original
        for col in numeric_columns:
            df_copie[col] = df_imputed[col]

    # Pour les colonnes de date, aucune imputation nécessaire car déjà converties

    return df_copie


## Fonction qui normalise le DataFrame:

def normalize_data(df, target_column='humidity_3days_after_date'):
    """
    Normalise les données numériques pour le machine learning
    
    Args:
        df: DataFrame pandas
        target_column: Nom de la colonne cible
    
    Returns:
        DataFrame normalisé et scaler pour la transformation inverse
    """
    df_copie = df.copy()
    
    # Identifier les colonnes numériques, sauf la cible
    numeric_columns = df_copie.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    # Normaliser les colonnes numériques
    scaler = StandardScaler()
    df_copie[numeric_columns] = scaler.fit_transform(df_copie[numeric_columns])
    
    return df_copie, scaler


## Fonction qui encode les variables catégorielles:

def encode_categorical_variables(df):
    """
    Encode les variables catégorielles pour le machine learning
    
    Args:
        df: DataFrame pandas avec variables catégorielles
    
    Returns:
        DataFrame avec variables catégorielles encodées et dictionnaire des encodeurs
    """
    df_copie = df.copy()
    
    # Identifier les colonnes catégorielles (texte)
    categoric_columns = df_copie.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Dictionnaire pour stocker les encodeurs
    encoders = {}

    all_months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    all_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    
    predefined_categories = {
        'Month': all_months,
        'Season': all_seasons
    }
    # Pour chaque colonne catégorielle
    for col in categoric_columns:
        # Pour les colonnes avec peu de valeurs uniques, utiliser one-hot encoding
        if df_copie[col].nunique() < 10 or col in predefined_categories:
            # Créer les variables dummy
            if col in predefined_categories:
                # Utiliser TOUTES les catégories prédéfinies (tous les mois/saisons)
                categories = predefined_categories[col]
            else:
                # Pour les autres colonnes, utiliser les catégories présentes
                categories = sorted(df_copie[col].unique())
            
            # Créer les variables dummy avec toutes les catégories possibles
            dummies = pd.get_dummies(df_copie[col], prefix=col, drop_first=True)
            
            # S'assurer que TOUTES les colonnes attendues sont présentes
            expected_columns = [f"{col}_{cat}" for cat in categories[1:]]  # drop_first=True
            
            for expected_col in expected_columns:
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            
            # Garder seulement les colonnes attendues dans le bon ordre
            dummies = dummies.reindex(columns=expected_columns, fill_value=0)
           
            # Ajouter les variables dummy au DataFrame
            df_copie = pd.concat([df_copie, dummies], axis=1)
           
            # Stocker l'information sur l'encodage
            encoders[col] = {'type': 'one-hot', 'categories': categories}
           
            # Supprimer la colonne originale
            df_copie = df_copie.drop(col, axis=1)
            
        else:
            # Pour les colonnes avec beaucoup de valeurs uniques, utiliser label encoding
            le = LabelEncoder()
            df_copie[col] = le.fit_transform(df_copie[col])
            encoders[col] = {'type': 'label', 'encoder': le}
    
    # Convertir les colonnes booléennes en float
    bool_cols = df_copie.select_dtypes(include=['bool']).columns
    df_copie[bool_cols] = df_copie[bool_cols].astype(float)
   
    return df_copie, encoders
# Fonction qui prépare les inputs X et leur label y

def prepare_data_for_model(df, target_column='humidity_3days_after_date'):
    """
    Prépare les données pour l'entraînement du modèle
    
    Args:
        df: DataFrame pandas prétraité
        target_column: Nom de la colonne cible
        test_size: Proportion des données pour le test
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    #Supprimer les colonnes qui ne sont pas utiles pour la prédiction
    colonnes_a_supprimer = [
        'id', 'end_date'  # Identifiants et dates brutes
    ]
    
    # # Supprimer uniquement les colonnes qui existent
    colonnes_a_supprimer = [col for col in colonnes_a_supprimer if col in df.columns]
    
    # # Supprimer également les colonnes de dates brutes
    
    
    # Créer X et y
    

    X = df.drop(colonnes_a_supprimer + [target_column], axis=1, errors='ignore')
    y = df[target_column]
    
    # Diviser les données en respectant l'ordre chronologique
    # On utilise les données les plus récentes pour le test
    
    
    return X, y


# Pipeline de préprocessing complet, 
def complete_pipeline(path, target_column='humidity_3days_after_date'):
    """
    Pipeline complet pour le prétraitement des données
    
    Args:
        chemin_fichier: Chemin vers le fichier de données
        target_column: Nom de la colonne cible
        test_size: Proportion des données pour le test
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # 1. Charger les données
    df = load_data(path)
    # 2. Extraire les caractéristiques temporelles
    df = add_month_season(df, 'end_date')
    # 3. Calculer les caractéristiques dérivées
    df = add_derivate_features(df)
    # 4. Gérer les valeurs manquantes
    df = generate_missing_values(df, target_column)
    #5. Encoder les variables catégorielles
    df_encode, encoders = encode_categorical_variables(df)
    print(encoders)
    # 6. Normaliser les données
    df_normalise, scaler = normalize_data(df_encode, target_column)
    # 6. Préparer les données pour le ML
    X,y = prepare_data_for_model(df_normalise, target_column)
    print('colonnes',X.columns)
    print(X,y)
    return X,y, scaler,encoders, df



# Fonction pour créer des séquences temporelles (utilisé ensuite par le LSTM)
def create_sequences(X, y, timesteps=3):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X.iloc[i:i+timesteps].values)  # Transformer en numpy array
        y_seq.append(y[i+timesteps])  # Cible correspondante
    return np.array(X_seq), np.array(y_seq)


## Fonction du pipeline de preprocessing qui prépare aussi les données d'entrainement et de test 
def complete_pipeline_with_sequences(path,target_column='humidity_3days_after_date',test_size=0.2):
    X,y, scaler,encoders, df=complete_pipeline(path,target_column)
    X_seq, y_seq=create_sequences(X, y)
    X_train, y_train,X_test, y_test = train_test_split(X_seq, y_seq, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler,encoders, df












