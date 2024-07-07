from tensorflow.keras.models import load_model
import numpy as np
import scipy.io.wavfile as wav
import speechpy
import os


def extract_features(file_path):
    fs, signal = wav.read(file_path)
    mfec_features = speechpy.feature.mfcc(signal, fs, num_cepstral=40)
    mfec_features = speechpy.processing.cmvnw(mfec_features, win_size=301, variance_normalization=True)
    return mfec_features


loaded_model = load_model('../models/speaker_verification_3D.keras')
print('model loaded')


def prepare_audio(file_path):
    features = extract_features(file_path)
    if features.shape[0] >= 80:
        features = features[:80, :]
    else:
        # Si hay menos de 80 frames, rellena con ceros
        padding = np.zeros((80 - features.shape[0], features.shape[1]))
        features = np.vstack((features, padding))
    features = np.expand_dims(features, axis=-1)  # Agrega una dimensión para el canal
    features = np.expand_dims(features, axis=0)  # Agrega una dimensión adicional para zeta
    features = np.expand_dims(features, axis=0)  # Agrega una dimensión para el batch
    return features


test_path = 'test_recordings'
for audio_path in os.listdir(test_path):
    if not audio_path.endswith('.wav'):
        continue
    full_audio_path = os.path.join(test_path, audio_path)
    print(full_audio_path)
    features = prepare_audio(full_audio_path)

    prediction = loaded_model.predict(features)

    print(f'Predicción: {prediction}')
    print(f'Clase predicha: {np.argmax(prediction)}')
