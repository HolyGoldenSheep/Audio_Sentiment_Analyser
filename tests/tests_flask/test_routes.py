def test_login_page(client):
    response = client.get("/login")
    assert response.status_code == 200


def test_dashboard_requires_login(client):
    response = client.get("/")
    assert response.status_code == 302