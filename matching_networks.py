import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from training.dataset import extract_features
import pickle
from sklearn.svm import OneClassSVM


def train_one_class_svm(support_set, nu=0.5):
    oc_svm = OneClassSVM(kernel='poly', gamma='auto', nu=nu)
    oc_svm.fit(support_set)

    return oc_svm


def load_support_set(support_set_file_path):

    with open(support_set_file_path, 'rb') as file:
        support_set, labels = pickle.load(file)
    return support_set, labels


def save_support_set(support_set, labels, support_set_file_path):

    with open(support_set_file_path, 'wb') as file:
        pickle.dump((support_set, labels), file)


def add_new_classes(new_dataset_path, embedding_model, support_set_file_path):

    support_set, labels = load_support_set(support_set_file_path)

    support_set = list(support_set)
    labels = list(labels)

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

            support_set.extend(class_embeddings)
            labels.extend(
                [class_name] * len(class_embeddings))

    if len(support_set) > 0:
        support_set = np.vstack(support_set)

    save_support_set(support_set, labels, support_set_file_path)
    oc_svm = train_one_class_svm(support_set)
    return support_set, labels, oc_svm


def speaker_verification(audio_file, embedding_model, support_set_file_path, oc_svm, label_map):
    support_set, labels = load_support_set(support_set_file_path)

    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)

    new_audio_embedding = embedding_model.predict(features)

    new_audio_embedding = np.array(new_audio_embedding)
    outlier_prediction = oc_svm.predict(new_audio_embedding.reshape(1, -1))

    if outlier_prediction == -1:
        return "unknown", None

    similarities = cosine_similarity(new_audio_embedding, support_set)

    predicted_index = np.argmax(similarities)
    predicted_class = labels[predicted_index]
    if isinstance(predicted_class, np.int64):
        label_map_inverted = {v: k for k, v in label_map.items()}
        predicted_class = label_map_inverted[predicted_class]

    return predicted_class, similarities
