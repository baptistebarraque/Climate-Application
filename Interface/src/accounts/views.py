from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout as auth_logout
from .forms import SignUpForm
from django.contrib import messages
import requests


# def verify_recaptcha(request):
#     recaptcha_response = request.POST.get('g-recaptcha-response')
#     secret_key = '6LezRJAqAAAAADZ1GFqoCmF8GSZo4TBBwwo8Rh4L'
#     url = 'https://www.google.com/recaptcha/api/siteverify'
#     payload = {
#         'secret': secret_key,
#         'response': recaptcha_response
#     }
#     response = requests.post(url, data=payload)
#     result = response.json()
#     return result.get('success', False)
def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            # if verify_recaptcha(request):
                user = form.save()
                login(request, user)  # Connecte automatiquement l'utilisateur
                messages.success(request, 'Compte créé avec succès !')
                return redirect('cartes')  # Redirige vers la page d'accueil ou autre
            # else:
            #     messages.error(request, 'CAPTCHA invalide.')

    else:
        form = SignUpForm()
    return render(request, 'accounts/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Bienvenue, {username}!')
                return redirect('cartes')  # Redirige vers la page d'accueil
            else:
                messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    else:
        form = AuthenticationForm()
    return render(request, 'accounts/login.html', {'form': form})

def logout_view(request):
    auth_logout(request)
    messages.info(request, "Vous êtes maintenant déconnecté.")
    return redirect('cartes')  # Redirige vers la page d'accueil ou de connexion

