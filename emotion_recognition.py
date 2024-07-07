import torch
import librosa

emotion_map = {
    "SAD": 0,
    "ANG": 1,
    "DIS": 2,
    "FEA": 3,
    "HAP": 4,
    "NEU": 5
}


def predict_emotion(audio_path, model, processor):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion_map_inv = {v: k for k, v in emotion_map.items()}
    predicted_emotion = emotion_map_inv[predicted_id]

    return predicted_emotion



