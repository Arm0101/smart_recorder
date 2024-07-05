import os
import numpy as np
import librosa
import noisereduce as nr
from utils import convert_audios_to_wav, preprocess_recording


def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    reduced_noise_audio = nr.reduce_noise(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=reduced_noise_audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    print(file_path)
    return mfccs_mean


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


def build_dataset(base_path, output_path):
    convert_audios_to_wav(base_path)
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        if os.path.isdir(person_path):
            for file_name in os.listdir(person_path):
                file_path = os.path.join(person_path, file_name)
                if file_name.endswith('.wav'):
                    output_dir = os.path.join(output_path, person_name)
                    preprocess_recording(file_path, output_dir=output_dir)


