import pytest
from pathlib import Path

@pytest.mark.asyncio
async def test_predict_requires_auth(client):
    response = await client.post("/predict")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_predict_success(client):
    # 1. Login
    login = await client.post(
        "/auth/token",
        data={"username": "test_user_pytest", "password": "password123"}
    )
    token = login.json()["access_token"]

    # 2. Audio file
    audio_path = Path("tests/assets/test.wav")
    assert audio_path.exists()

    with open(audio_path, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = await client.post(
            "/predict",
            headers={"Authorization": f"Bearer {token}"},
            files=files
        )

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
