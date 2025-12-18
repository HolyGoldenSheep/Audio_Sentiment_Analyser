import pytest

@pytest.mark.asyncio
async def test_signup(client):
    payload = {
        "username": "test_user_pytest",
        "password": "password123",
        "email": "test@test.com"
    }

    response = await client.post("/auth/signup", json=payload)
    assert response.status_code in (200, 400)  


@pytest.mark.asyncio
async def test_login(client):
    data = {
        "username": "test_user_pytest",
        "password": "password123"
    }

    response = await client.post("/auth/token", data=data)
    assert response.status_code == 200

    token = response.json()
    assert "access_token" in token
