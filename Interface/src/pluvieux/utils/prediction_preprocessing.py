import re
import pandas as pd
from datetime import datetime, timedelta
import datetime as dt
import csv
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
import warnings
from .preprocessing import *


def load_data(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
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
def generate_missing_values_pred(df):
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


def normalize_data_pred(df):
    """
    Normalise les données numériques pour le machine learning, 
    en excluant les colonnes constantes de la normalisation.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        df_copie: DataFrame avec colonnes normalisées
        scaler: StandardScaler entraîné (pour inverse_transform éventuel)
        cols_scaled: Liste des colonnes qui ont été normalisées
    """
    df_copie = df.copy()

    # Identifier les colonnes numériques
    numeric_columns = df_copie.select_dtypes(include=[np.number]).columns.tolist()

    # Détecter les colonnes avec variance non nulle
    cols_scaled = [col for col in numeric_columns if df_copie[col].std() > 0]

    # Appliquer StandardScaler uniquement sur les colonnes non constantes
    scaler = StandardScaler()
    df_copie[numeric_columns] = scaler.fit_transform(df_copie[numeric_columns])

    return df_copie, scaler, cols_scaled

def prepare_data_for_prediction(df):
    """
    Prépare les données pour la prédiction (sans colonne cible)
    """
    # Colonnes à supprimer
    colonnes_a_supprimer = [
        'id', 'end_date'  # Identifiants et dates brutes
    ]
    
    # Supprimer uniquement les colonnes qui existent
    colonnes_a_supprimer = [col for col in colonnes_a_supprimer if col in df.columns]
    
    # Créer X (toutes les caractéristiques à utiliser pour la prédiction)
    X = df.drop(colonnes_a_supprimer, axis=1, errors='ignore')
    
    return X



def create_sequences_for_prediction(X, timesteps=3):
    """
    Crée des séquences temporelles pour la prédiction
    """
    X_seq = []
    for i in range(len(X) - timesteps + 1):
        X_seq.append(X.iloc[i:i+timesteps].values)
    return np.array(X_seq)

def preprocessing_pipeline_for_prediction(path):
    """
    Pipeline complet pour le prétraitement des données de prédiction
    """
    # 1. Charger les données
    
    df = load_data(path)
    print(df)
    
    # 2. Extraire les caractéristiques temporelles
    df = add_month_season(df, 'end_date')
    print('2')
    print('la')
    # 3. Calculer les caractéristiques dérivées
    df = add_derivate_features(df)
    df=prepare_data_for_prediction(df)
    print('pala')
    print('3')
    # 4. Gérer les valeurs manquantes
    df = generate_missing_values_pred(df)
    print('4')
    # 5. Encoder les variables catégorielles
    df_encode, encoders = encode_categorical_variables(df)
    print(df_encode.columns)
    # 6. Normaliser les données
    df_normalise, scaler = normalize_data(df_encode)
    print(df_normalise)
    # 7. Préparer les données pour la prédiction
    X = prepare_data_for_prediction(df_normalise)
    
    print('colonnes prédiction:X', X.columns)
    # 8. Créer des séquences temporelles
    X_seq = create_sequences_for_prediction(X)
    print(X_seq)
    return X_seq, scaler, encoders, df

