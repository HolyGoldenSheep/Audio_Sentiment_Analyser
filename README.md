# Audio Sentiment Analyser
Présentation du projet

Audio Sentiment Analyser est une application permettant de détecter automatiquement les émotions présentes dans un fichier audio à l’aide d’un modèle d’intelligence artificielle.

Le système est conçu comme une architecture applicative complète comprenant :

une API d’inférence basée sur FastAPI

une interface web Flask

une base de données MongoDB

un système de monitoring avec Prometheus et Grafana

une chaîne CI/CD automatisée via GitHub Actions

un environnement conteneurisé avec Docker

L’objectif est de transformer un signal audio brut en information exploitable (émotion détectée et score de confiance) dans une architecture compatible avec une utilisation professionnelle.

# Architecture du projet

Le projet est organisé selon une architecture modulaire permettant de séparer les différentes responsabilités techniques.

Audio_Sentiment_Analyser/
│
├── app/                # API FastAPI (service d'inférence IA)
│   ├── main.py
│   ├── models/
│   ├── services/
│   └── utils/
│
├── webapp/             # Interface utilisateur Flask
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── tests/              # Tests automatisés (Pytest)
│
├── monitoring/         # Configuration Prometheus / Grafana
│
├── docker/             # Configuration Docker
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

Cette architecture permet :

de séparer l’inférence IA de l’interface utilisateur

de faciliter le déploiement et la maintenance

d’intégrer facilement des outils de supervision.

# Fonctionnement du système

Le fonctionnement global de l’application repose sur le flux suivant :

L’utilisateur se connecte à l’interface web.

Il téléverse un fichier audio.

L’interface envoie le fichier à l’API via une requête HTTP sécurisée.

L’API traite le fichier audio et applique le modèle d’intelligence artificielle.

L’émotion détectée et le score de confiance sont retournés à l’utilisateur.

Les résultats sont enregistrés dans la base de données MongoDB.

# Technologies utilisées
Backend API

FastAPI

Python

Motor (MongoDB Async Driver)

JWT Authentication

Machine Learning

TensorFlow

Wav2Vec2

Librosa

NumPy

Interface utilisateur

Flask

HTML / Jinja2

CSS

Monitoring

Prometheus

Grafana

Flask Monitoring Dashboard

Infrastructure

Docker

Docker Compose

GitHub Actions (CI/CD)

# Installation du projet
Prérequis

Avant d’installer le projet, il est nécessaire d’avoir :

Python 3.10+

pip

MongoDB

Docker (optionnel mais recommandé)

Installation en environnement local

Créer un environnement virtuel :

python -m venv .venv

Activer l’environnement :

Linux / Mac

source .venv/bin/activate

Windows

.venv\Scripts\activate

Installer les dépendances :

pip install -r requirements.txt
Configuration des variables d’environnement

Créer un fichier .env à la racine du projet.

Exemple :

SECRET_KEY=your_secret_key
MONGO_URI=mongodb://localhost:27017/audio_sentiment
JWT_SECRET=your_jwt_secret
API_BASE_URL=http://localhost:8000

Ces variables permettent de configurer :

la connexion à la base de données

la sécurité de l’application

l’authentification JWT.

Lancer l’API FastAPI

Démarrer le serveur :

uvicorn app.main:app --reload

L’API sera accessible à l’adresse :

http://localhost:8000

Documentation interactive :

http://localhost:8000/docs
Lancer l’interface web

Depuis le dossier webapp :

python app.py

L’application web sera accessible sur :

http://localhost:5000
Utilisation de l’API
Authentification

Endpoint :

POST /login

Permet d’obtenir un token JWT.

Exemple de réponse :

{
  "access_token": "...",
  "token_type": "bearer"
}

Ce token doit être utilisé dans l’en-tête des requêtes :

Authorization: Bearer <token>
Prédiction d’émotion

Endpoint :

POST /predict

Paramètres :

fichier audio (multipart)

Réponse :

{
  "emotion": "happy",
  "confidence": 0.87
}
Tests automatisés

Les tests sont réalisés avec Pytest.

Lancer les tests :

pytest

Les tests vérifient :

l’authentification

la prédiction

les erreurs d’accès

la validation des fichiers audio.

Intégration continue (CI)

Le projet utilise GitHub Actions pour automatiser :

l’installation de l’environnement

l’exécution des tests

la construction de l’image Docker

Le pipeline CI se déclenche automatiquement lors :

d’un push

d’une pull request

Conteneurisation avec Docker

Construire l’image Docker :

docker build -t audio-sentiment-api .

Lancer les services avec Docker Compose :

docker-compose up --build

Les services démarrés :

API FastAPI

Interface web

MongoDB

Prometheus

Grafana

Monitoring de l’application

Le système de supervision repose sur plusieurs outils.

Prometheus

Collecte les métriques :

temps de réponse des requêtes

nombre de requêtes

erreurs HTTP

performances du modèle

Grafana

Permet de visualiser les métriques sous forme de dashboards.

Adresse par défaut :

http://localhost:3000
Flask Monitoring Dashboard

Permet de surveiller :

les endpoints

le temps de réponse

les exceptions.

Sécurité

L’application intègre plusieurs mécanismes de sécurité :

authentification JWT

validation du type MIME des fichiers

limitation de taille des fichiers uploadés

gestion des erreurs

contrôle d’accès aux endpoints protégés.

Monitoring du modèle (MLOps)

Le système implémente une feedback loop.

Après chaque prédiction :

l’utilisateur peut confirmer ou corriger le résultat

les données sont enregistrées dans MongoDB

ces données peuvent servir à réentraîner le modèle.

Cette approche permet d’améliorer progressivement la performance du modèle.

Contribution

Pour contribuer au projet :

Fork du dépôt

Création d’une branche

Implémentation des modifications

Création d’une Pull Request

Auteur

Karl Benton
Développeur IA

Licence

Projet réalisé dans le cadre du Titre Professionnel Développeur en Intelligence Artificielle (RNCP 37827).
