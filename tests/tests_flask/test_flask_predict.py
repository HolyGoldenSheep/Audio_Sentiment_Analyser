def test_predict_requires_login(client):
    response = client.post("/predict")
    assert response.status_code == 302