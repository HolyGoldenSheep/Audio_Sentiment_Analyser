import pytest

@pytest.mark.asyncio
async def test_predict_rejects_large_file(client):

    # First sign up
    await client.post("/auth/signup", json={
        "username": "securityuser",
        "password": "StrongPass123",
        "email": "sec@test.com"
    })

    # Then login
    login = await client.post("/auth/token", data={
        "username": "securityuser",
        "password": "StrongPass123"
    })

    token = login.json()["access_token"]

    fake_audio = b"0" * (25 * 1024 * 1024)  # 25MB > 20MB

    response = await client.post(
        "/predict",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("big.wav", fake_audio, "audio/wav")}
    )

    assert response.status_code in (400, 413)
