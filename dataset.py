from utils import ogg_to_wav
import os
import numpy as np
import librosa


def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


def convert_audios_to_wav(path):
    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)
        if os.path.isdir(person_path):
            for file_name in os.listdir(person_path):
                if file_name.endswith('.ogg'):
                    file_path = os.path.join(person_path, file_name)
                    output_path = os.path.splitext(file_path)[0] + '.wav'
                    if not os.path.exists(output_path):
                        ogg_to_wav(file_path, output_path)
                    os.remove(file_path)


def prepare_dataset(base_path):
    data = []
    labels = []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)

        if os.path.isdir(person_path):
            for file_name in os.listdir(person_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(person_path, file_name)
                    features = extract_features(file_path)
                    data.append(features)
                    if person_name not in label_map:
                        label_map[person_name] = label_count
                        label_count += 1
                    labels.append(label_map[person_name])

    data = np.array(data)
    labels = np.array(labels)
    return data, labels, label_map
