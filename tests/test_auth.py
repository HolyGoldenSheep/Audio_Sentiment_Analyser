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

    # create the user
    await client.post("/auth/signup", json={
        "username": "test_user_pytest",
        "password": "password123",
        "email": "test_user_pytest@example.com"
    })

    # login
    response = await client.post(
        "/auth/token",
        data={
            "username": "test_user_pytest",
            "password": "password123"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
