import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from training.dataset import extract_features

model = load_model('../models/speaker_verification_hyb_cnn.keras')


def prediction(audio_path):
    n_mfcc = 40
    max_len = 500
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # Hacer la predicción
    prediction = model.predict(mfcc)
    predicted_speaker = np.argmax(prediction)
    print(f"Prediction: {prediction}")
    print(f"Predicted Speaker: {predicted_speaker}")


test_path = 'test_recordings'
for file in os.listdir(test_path):
    if not file.endswith('.wav'):
        continue

    file_path = os.path.join(test_path, file)
    print(f"Processing file: {file_path}")
    prediction(file_path)
