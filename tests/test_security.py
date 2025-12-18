import pytest

@pytest.mark.asyncio
async def test_predict_rejects_large_file(client):
    fake_audio = b"0" * (15 * 1024 * 1024) 

    response = await client.post(
        "/predict",
        files={"file": ("big.wav", fake_audio, "audio/wav")}
    )

    assert response.status_code in (400, 413)
