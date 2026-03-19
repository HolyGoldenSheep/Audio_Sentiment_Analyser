# Audio Sentiment Analyser

API et application web permettant de **détecter les émotions dans un fichier audio** à l’aide d’un modèle d’intelligence artificielle.

Le projet met en place une **architecture complète d'application IA** incluant :

- une **API d’inférence FastAPI**
- une **interface utilisateur Flask**
- une **base de données MongoDB**
- un **monitoring Prometheus + Grafana**
- une **pipeline CI/CD avec GitHub Actions**
- une **infrastructure conteneurisée Docker**

Ce projet s’inscrit dans une démarche **MLOps** visant à industrialiser le cycle de vie d’une application d’intelligence artificielle.

---

# Architecture du projet
Audio_Sentiment_Analyser/

app/ # API FastAPI (service d'inférence IA)
- main.py
- models/
- services/
- utils/

   webapp/ # Interface utilisateur Flask
- app.py
- templates/
- static/

   tests/ # Tests automatisés (Pytest)

   monitoring/ # Configuration Prometheus / Grafana

   docker/ # Configuration Docker

   docker-compose.yml
   Dockerfile
   requirements.txt
  README.md

Cette architecture permet :

- une **séparation claire entre l’API et l’interface**
- une **meilleure maintenabilité**
- une **facilité de déploiement**
- l’intégration d’outils de **supervision et monitoring**

---

#  Technologies utilisées

## Backend API

- FastAPI
- Python
- MongoDB (Motor async driver)
- JWT Authentication

## Machine Learning

- TensorFlow
- Wav2Vec2
- Librosa
- NumPy

## Interface utilisateur

- Flask
- HTML / Jinja2
- CSS

## Monitoring

- Prometheus
- Grafana
- Flask Monitoring Dashboard

## Infrastructure

- Docker
- Docker Compose
- GitHub Actions (CI/CD)

---

#  Fonctionnement de l’application

Le système fonctionne selon le flux suivant :

1. L’utilisateur se connecte à l’interface web.
2. Il téléverse un fichier audio.
3. L’interface envoie le fichier à l’API via une requête HTTP sécurisée.
4. L’API traite le fichier audio et applique le modèle d’intelligence artificielle.
5. L’émotion détectée et le score de confiance sont retournés à l’utilisateur.
6. Les résultats sont enregistrés dans la base de données MongoDB.

---

#  Installation du projet

## Prérequis

Avant d’installer le projet :

- Python 3.10+
- pip
- MongoDB
- Docker (optionnel mais recommandé)

---

## Installation locale

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l’environnement :

Linux / Mac

source .venv/bin/activate

Windows

.venv\Scripts\activate

Installer les dépendances :

pip install -r requirements.txt

# Configuration

Créer un fichier .env à la racine du projet.

Exemple :

```
SECRET\_KEY=your\_secret\_key
JWT\_SECRET=your\_jwt\_secret
MONGO\_URI=mongodb://localhost:27017/audio\_sentiment
API\_BASE\_URL=http://localhost:8000
```
# Lancer l’API FastAPI

Démarrer le serveur :

uvicorn app.main:app --reload

API disponible sur :

http://localhost:8000

Documentation interactive :

http://localhost:8000/docs

# Lancer l’interface web

Depuis le dossier webapp :

python app.py

Interface accessible sur :

http://localhost:5000

# Utilisation de l’API
Authentification

Endpoint :

POST /login

Réponse :

{
  "access_token": "token",
  "token_type": "bearer"
}

Le token doit être utilisé dans les requêtes :

Authorization: Bearer <token>
# Prédiction d’émotion

Endpoint :

POST /predict

Paramètre :

fichier audio (multipart)

Réponse :

{
  "emotion": "happy",
  "confidence": 0.87
}
# Tests automatisés

Les tests sont réalisés avec Pytest.

Lancer les tests :

pytest

Les tests couvrent :

- authentification
- prédiction
- validation des fichiers
- gestion des erreurs

# Intégration continue (CI)

Le projet utilise GitHub Actions pour automatiser :

- l’installation de l’environnement
- l’exécution des tests
- la construction de l’image Docker

Le pipeline se déclenche automatiquement lors :

- d’un push
- d’une pull request

# Conteneurisation Docker

Construire l’image :

docker build -t audio-sentiment-api .

Lancer les services :

docker-compose up --build

Services démarrés :

- API FastAPI
- Interface web
- MongoDB
- Prometheus
- Grafana
- 
# Monitoring

Le système de supervision repose sur plusieurs outils.

Prometheus

Collecte les métriques :

- nombre de requêtes
- latence
- erreurs HTTP
- performances du modèle

Grafana

Permet de visualiser les métriques sous forme de dashboards.
Adresse par défaut :

http://localhost:3000

Flask Monitoring Dashboard

Permet de surveiller :
- les endpoints
- le temps de réponse
- les exceptions

# Sécurité

Plusieurs mécanismes sont implémentés :

- authentification JWT
- validation des fichiers uploadés
- contrôle d’accès aux endpoints
- gestion des erreurs

# MLOps et amélioration continue

Le projet intègre une feedback loop.

Après chaque prédiction :

- l’utilisateur peut valider ou corriger le résultat
- les données sont enregistrées dans MongoDB
- es données peuvent être utilisées pour réentraîner le modèle

Cela permet d’améliorer progressivement les performances du modèle.

# Auteur
Karl Benton
Projet réalisé dans le cadre du Titre Professionnel Développeur en Intelligence Artificielle (RNCP 37827).

# Licence
Projet pédagogique réalisé dans le cadre de la formation développeur IA.
