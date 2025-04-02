# projet-900M2

Le répertoire archives contient le rendu des éléves au format brut

## Avancement

* L'affichage des cartes supperposées avec slider fonctionne
* Le projet google earth engine est en dur

## Définition d'une carte

 * Une carte (pmap) est composée de trois layers 
 * Une couche (layer) est composé d'une datasource et d'une couleur
 * Une couleur (color) est composé de la couleur médiane du dégradé, 
   de la couleur minimale et de la couleur maximale, ainsi que du nombre 
   de couleurs dans le dégradé
 * Une source de données (datasource) correspond dans Google Earth Engine à :
   * Une collection
   * Une band
   * La valeur minimale
   * La valeur maximale
   * Le pas de temps en jours

## Utilisation (debug)

 * installer un environnment virutel python
 * installer les dépendance (requirements.txt)
 * lancer django
     * Préparer et effectuer les migrations
     * Collecter les fichiers statiques

```bash
python -m venv /path/to/new/virtual/environment
. /path/to/new/virtual/environment/bin/activate
pip install -f requirements.txt
python manage.py makemigrations 
python manage.py makemigrations pluvieux
python manage.py migrate
python manage.py migrate pluvieux
python manage.py collectstatic
python manage.py runserver
```
Pour travailler en local il faut créer un fichier settings_dev avec des hôtes locaux et rajouter à la fin des fichiers activate et activate.bat de l'environnement virtuel les lignes suivantes:
* export DJANGO_SETTINGS_MODULE="pluvieux.settings_dev"
* export DJANGO_READ_LOCAL_SETTINGS=True
Il faut aussi modifier le fichier wsgi.py en remplaçant settings.py par settings_dev.py
Une fois que le travail en local a été réalisé il faut remettre le fichier settings.py en paramètre pour permettre le merge sur la branche main.
