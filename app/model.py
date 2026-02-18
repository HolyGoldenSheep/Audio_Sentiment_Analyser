# model.py
import torch
import torchaudio
import numpy as np
import soundfile as sf
from io import BytesIO
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import tensorflow as tf

device = torch.device("cpu") 
torch.set_num_threads(1)

class SentimentModel:

    EMOTION_LABELS = [
        "neutral", "calm", "happy", "sad",
        "angry", "fear", "disgust", "surprise"
    ]

    def __init__(self, classifier_path: str):
        print("Loading Wav2Vec2 (facebook/wav2vec2-base)...")

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

        base_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        base_model.eval()

        print("Applying dynamic quantization (int8)...")

        self.wav2vec = torch.quantization.quantize_dynamic(
            base_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        self.wav2vec.to(device)

        # Free original model memory
        del base_model

        print("Loading classifier as TensorFlow SavedModel:", classifier_path)
        self.classifier = tf.saved_model.load(classifier_path)
        self.infer = self.classifier.signatures["serving_default"]

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=16000
        )

    def _load_audio_from_bytes(self, audio_bytes: bytes):
        """
        Load WAV audio using soundfile (no FFmpeg, no torchcodec).
        Returns: waveform (torch.Tensor), sample_rate
        """
        data, sr = sf.read(BytesIO(audio_bytes), dtype="float32")

        # Convert to torch tensor
        waveform = torch.from_numpy(data)

        # If stereo → mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        # Shape: [1, time]
        waveform = waveform.unsqueeze(0)

        return waveform, sr

    def _extract_embedding(self, audio_bytes: bytes):
        waveform, sr = self._load_audio_from_bytes(audio_bytes)

        # Resample if needed
        if sr != 16000:
            waveform = self.resampler(waveform)

        inputs = self.processor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(device)

        with torch.no_grad():
            outputs = self.wav2vec(inputs)
            hidden_states = outputs.last_hidden_state

        # Mean pooling
        embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def predict_bytes(self, audio_bytes: bytes):
        embedding = self._extract_embedding(audio_bytes)
        embedding = np.expand_dims(embedding, axis=0).astype(np.float32)

        tensor = tf.constant(embedding)
        preds_dict = self.infer(tensor)
        preds_array = list(preds_dict.values())[0].numpy()

        label_id = int(np.argmax(preds_array))
        confidence = float(np.max(preds_array))

        return {
            "label": self.EMOTION_LABELS[label_id],
            "confidence": confidence
        }

    @classmethod
    def load(cls, path: str):
        return cls(classifier_path=path)
