import numpy as np
from training.dataset import extract_features
from utils import load_label_map


def speaker_verification(audio_file, classification_model, label_map_file):
    label_map = load_label_map(label_map_file)
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    prediction = classification_model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    label_map_inverted = {v: k for k, v in label_map.items()}
    predicted_person = label_map_inverted[predicted_label[0]]
    return prediction, predicted_label[0], predicted_person
