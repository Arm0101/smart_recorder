import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa

model_path = "models/emo_model"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)
emotion_map = {
    "SAD": 0,
    "ANG": 1,
    "DIS": 2,
    "FEA": 3,
    "HAP": 4,
    "NEU": 5
}


def predict_emotion(audio_path):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Hacer la predicción
    with torch.no_grad():
        logits = model(**inputs).logits

    # Obtener la etiqueta predicha
    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion_map_inv = {v: k for k, v in emotion_map.items()}
    predicted_emotion = emotion_map_inv[predicted_id]

    return predicted_emotion



audio_path = "test_emotions/jose03.wav"

# Realizar la predicción
predicted_emotion = predict_emotion(audio_path)
print(f"La emoción predicha es: {predicted_emotion}")
