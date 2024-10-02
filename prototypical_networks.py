import os
from training.dataset import extract_features
import numpy as np
import pickle


def calculate_prototypes(embeddings, labels, label_map):
    label_map_inverted = {v: k for k, v in label_map.items()}
    prototypes = {}
    for label in np.unique(labels):
        class_embeddings = embeddings[labels == label]
        class_name = label_map_inverted[label]
        prototypes[class_name] = np.mean(class_embeddings, axis=0)

    return prototypes


def load_prototypes(prototypes_file_path):
    with open(prototypes_file_path, 'rb') as file:
        prototypes = pickle.load(file)
    return prototypes


def save_prototypes(prototypes, prototypes_file_path):
    with open(prototypes_file_path, 'wb') as file:
        pickle.dump(prototypes, file)


def add_new_classes(new_dataset_path, embedding_model, prototypes_file_path):
    prototypes = load_prototypes(prototypes_file_path)

    for class_name in os.listdir(new_dataset_path):
        class_path = os.path.join(new_dataset_path, class_name)
        if os.path.isdir(class_path):
            class_embeddings = []

            for audio_file in os.listdir(class_path):
                audio_file_path = os.path.join(class_path, audio_file)
                features = extract_features(audio_file_path)
                features = np.expand_dims(features, axis=0)
                features = np.expand_dims(features, axis=-1)

                embedding = embedding_model.predict(features)
                class_embeddings.append(embedding)
            new_class_prototype = np.mean(class_embeddings, axis=0)
            prototypes[class_name] = new_class_prototype
    save_prototypes(prototypes, prototypes_file_path)
    return prototypes


def speaker_verification(audio_file, embedding_model, prototypes_file_path, label_map):
    prototypes = load_prototypes(prototypes_file_path)

    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)

    new_audio_embedding = embedding_model.predict(features)
    distances = {label: np.linalg.norm(new_audio_embedding - proto) for label, proto in prototypes.items()}

    predicted_class = min(distances, key=distances.get)
    if isinstance(predicted_class, np.int16):
        label_map_inverted = {v: k for k, v in label_map.items()}
        predicted_class = label_map_inverted[predicted_class]
    return predicted_class, distances
