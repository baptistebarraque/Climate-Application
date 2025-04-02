from .settings import *

# Paramètres spécifiques au développement local
DEBUG = True  # Garde le mode débogage activé en développement

# Permet l'accès à ton serveur local
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

# Base de données pour l'environnement local
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # Utilisation de SQLite en local
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Paramètres CSRF pour le développement local
CSRF_TRUSTED_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"]

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Désactivez les paramètres de sécurité qui pourraient gêner le développement local
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
