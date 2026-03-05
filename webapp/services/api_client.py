import requests
from flask import session
from config import Config

class APIClient:

    @staticmethod
    def login(username, password):
        response = requests.post(
            f"{Config.API_BASE_URL}/auth/token",
            data={
                "username": username,
                "password": password
            },
            timeout=10
        )
        return response

    @staticmethod
    def register(username, email, password):
        response = requests.post(
            f"{Config.API_BASE_URL}/auth/signup",
            json={
                "username": username,
                "email": email,
                "password": password
            }
        )
        return response

    @staticmethod
    def predict(file):
        token = session.get("access_token")

        headers = {
            "Authorization": f"Bearer {token}"
        }

        files = {
            "file": (file.filename, file.stream, file.mimetype)
        }

        response = requests.post(
            f"{Config.API_BASE_URL}/predict",
            headers=headers,
            files=files
        )

        return response