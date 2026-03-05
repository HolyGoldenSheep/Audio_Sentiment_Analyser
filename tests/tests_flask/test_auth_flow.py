def test_register_route(client):
    response = client.get("/register")
    assert response.status_code == 200