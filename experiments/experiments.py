import os
import tensorflow as tf
from prototypical_networks import speaker_verification, add_new_classes
from utils import load_label_map

if __name__ == '__main__':
    path = '../test/test_recordings'
    label_map_file = '../models/speaker_identification_label_map.json'
    label_map = load_label_map(label_map_file)
    prototypes_file_path = '../models/speaker_prototypes.pkl'
    classification_model = tf.keras.models.load_model('../models/speaker_identification.keras')
    embedding_model = tf.keras.models.load_model('../models/speaker_identification_embedding.keras')
    add_new_classes('../new_dataset/', embedding_model, prototypes_file_path)

    files = list(filter(lambda f: f.endswith('.wav'), os.listdir(path)))
    total = len(files)
    count_ovidio = 0
    err_ovidio = 0
    count_carlos = 0
    err_carlos = 0
    count_jose = 0
    err_jose = 0
    count_armando = 0
    err_armando = 0
    count_s1 = 0
    err_s1 = 0
    count_s2 = 0
    err_s2 = 0
    count_s3 = 0
    err_s3 = 0
    count_s4 = 0
    err_s4 = 0
    for file in files:
        audio = os.path.join(path, file)
        prediction, dist = speaker_verification(audio, embedding_model, prototypes_file_path, label_map)

        print(f'prediction {prediction}\n')
        print('--------------------------------------------------------------------------')
        if file.startswith('ovidio'):
            count_ovidio += 1
            if prediction != 'ovidio':
                err_ovidio += 1
        if file.startswith('carlos'):
            count_carlos += 1
            if prediction != 'carlos':
                err_carlos += 1
        elif file.startswith('jose'):
            count_jose += 1
            if prediction != 'jose':
                err_jose += 1
        elif file.startswith('armando'):
            count_armando += 1
            if prediction != 'armando':
                err_armando += 1
        elif file.startswith('speaker_1'):
            count_s1 += 1
            if prediction != 'speaker_1':
                err_s1 += 1
        elif file.startswith('speaker_2'):
            count_s2 += 1
            if prediction != 'speaker_2':
                err_s2 += 1
        elif file.startswith('speaker_3'):
            count_s3 += 1
            if prediction != 'speaker_3':
                err_s3 += 1
        elif file.startswith('speaker_4'):
            count_s4 += 1
            if prediction != 'speaker_4':
                err_s4 += 1

    print('ovidio', (err_ovidio / count_ovidio) * 100)
    print('jose', (err_jose / count_jose) * 100)
    print('armando', (err_armando / count_armando) * 100)
    print('carlos', (err_carlos / count_carlos) * 100)
    print('s1', (err_s1 / count_s1) * 100)
    print('s2', (err_s2 / count_s2) * 100)
    print('s3', (err_s3 / count_s3) * 100)
    print('s4', (err_s4 / count_s4) * 100)
