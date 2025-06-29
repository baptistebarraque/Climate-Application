import csv
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch, BayesianOptimization, Hyperband
import keras_tuner as kt

############ Entraînement du modèle #################




def tuning_model(X_train,y_train,X_test,y_test, max_trials=20,max_epochs=20, model_name='best_model'):
    print(len(X_train),len(y_train))
    def build_model(hp):
        model = Sequential([
            LSTM(
                hp.Int('units_1', min_value=32, max_value=128, step=16),
                return_sequences=True,
                input_shape=(3, X_train.shape[2])
            ),
            Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)),
            LSTM(
                hp.Int('units_2', min_value=32, max_value=128, step=16),
                return_sequences=False
            ),
            Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='mse',
            metrics=['mae']
        )

        return model
    tuner = RandomSearch(

    hypermodel=build_model,

    objective='val_mae',

    max_trials=max_trials,  # Ici, max_trials est correctement placé

    executions_per_trial=2,

    directory='my_tuning_dir',

    project_name='lstm_tuning',

    overwrite=True # Pour éviter les erreurs si le répertoire existe déjà

    )
    tuner.search(

    X_train,

    y_train,

    epochs=max_epochs,  # Nombre d'epochs pour chaque essai (peut être petit au début)

    validation_data=(X_test, y_test),

    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)] # Early stopping pour éviter le sur-apprentissage

    )
    model_name=model_name+'.h5'
    best_model = tuner.get_best_models(num_models=1)[0]
    #best_model.summary()

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    history = best_model.fit(X_train, y_train, epochs=130, validation_data=(X_test, y_test))
    model_path='trainedmodels/'+model_name
    best_model.save(model_path)
    best_model.evaluate(X_test, y_test)
    return best_model, history, best_hyperparameters

def model_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    smape = np.mean(np.abs(y_test - y_pred) / ((np.abs(y_test) + np.abs(y_pred)) / 2)) * 100
    medae = median_absolute_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)
    indicators={'mse':mse,'mae':mae,'r2':r2,'mape':mape,'smape':smape,'medae':medae,'explained_variance':explained_variance}

    # Afficher les résultats
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.4f}%")
    print(f"Median Absolute Error (MedAE): {medae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance Score: {explained_variance:.4f}")
    return mse, mae, r2, mape, smape, explained_variance

