import pytest
import numpy as np
from pathlib import Path
from app.model import SentimentModel

MODEL_PATH = "app/Model/emotion_model"
test_audio_path = Path("tests/assets/test.wav")

@pytest.fixture(scope="session")
def model():
    """
    Charge le model une fois pour le test 

    """
    return SentimentModel.load(MODEL_PATH)

@pytest.fixture(scope="session")
def audio_bytes():
    """
    Charge le fichier audio

    """
    assert test_audio_path.exists(), "Le fichier test.wav est manquant"
    with open(test_audio_path, "rb") as f:
        return f.read()
    
def test_audio_file_exists():
    """
    Vérifie que le fichier audio de test est present

    """
    assert test_audio_path.exists()

def test_load_audio_from_bytes(model, audio_bytes):
    """
    Verfie que l'audio est bien chargé, converti en mono si besoin et formaté au bon shape

    """
    waveform, sr = model._load_audio_from_bytes(audio_bytes)
    assert waveform is not None
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1 
    assert sr > 0

def test_extract_embedding(model, audio_bytes):
    """
    Verifie que l'embedding Wav2Vec est bien chargé

    """

    embedding = model._extract_embedding(audio_bytes)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0
    assert not np.isnan(embedding).any()

def test_predict_output_structure(model, audio_bytes):
    """
    verifie que la prediction retourne la structure demander

    """
    result = model.predict_bytes(audio_bytes)
    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result

def test_label_is_valid(model, audio_bytes):
    """
    Vérifie que le label appartient aux classes définies.
    """
    result = model.predict_bytes(audio_bytes)

    assert result["label"] in model.EMOTION_LABELS


def test_confidence_range(model, audio_bytes):
    """
    Vérifie que la confiance est comprise entre 0 et 1.
    """
    result = model.predict_bytes(audio_bytes)

    confidence = result["confidence"]

    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0