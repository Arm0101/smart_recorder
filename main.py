from utils import convert_audios_to_wav
from training.dataset import extract_features
import tensorflow as tf
import numpy as np
import os
import librosa


def load_model(model_save_path):
    if os.path.exists(model_save_path):
        loaded_model = tf.keras.models.load_model(model_save_path)
        return loaded_model
    else:
        return None


if __name__ == '__main__':
    labels = ['carlos', 'jose', 'armando', 'ovidio']

    # build_dataset('recordings', 'dataset')
    # print('dataset completed')
    test_paths = 'test/test_recordings'
    convert_audios_to_wav(test_paths)
    model = load_model('models/speaker_identification.keras')
    if not model:
        exit(1)
    for audio_path in os.listdir(test_paths):
        audio, sr = librosa.load(os.path.join(test_paths, audio_path))
        # if librosa.get_duration(y=audio, sr=sr) > 11:
        #     continue
        full_path = person_path = os.path.join(test_paths, audio_path)
        features = extract_features(full_path)

        features = np.expand_dims(features, axis=0)  # Añadir dimensión de lote
        features = np.expand_dims(features, axis=-1)  # Añadir dimensión de canal

        prediction = model.predict(features)
        print(f'prediction {prediction}\n')
        predicted_label = np.argmax(prediction, axis=1)
        print(f"Predicted label: {predicted_label}")
        print(labels[predicted_label[0]])
        print('-------------------------------\n')
        # label_map_inverted = {v: k for k, v in label_map.items()}  # Invertir el mapa de etiquetas
        # predicted_person = label_map_inverted[predicted_label[0]]

    # print(f"Predicted person: {predicted_person}")
