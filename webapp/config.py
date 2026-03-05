import os

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "flask-secret-dev")
    API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")