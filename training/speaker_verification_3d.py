import os
import numpy as np
import scipy.io.wavfile as wav
import speechpy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def extract_features(file_path):
    fs, signal = wav.read(file_path)
    mfec_features = speechpy.feature.mfcc(signal, fs, num_cepstral=40)
    mfec_features = speechpy.processing.cmvnw(mfec_features, win_size=301, variance_normalization=True)
    return mfec_features


# Ruta del dataset
dataset_path = '../dataset'

data = []
labels = []
speakers = ['jose', 'armando', 'ovidio']

for label, speaker in enumerate(speakers):
    speaker_path = os.path.join(dataset_path, speaker)
    for file_name in os.listdir(speaker_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(speaker_path, file_name)
            features = extract_features(file_path)
            print(file_path)
            if features.shape[0] >= 80:  # Asegúrate de tener al menos 80 frames
                data.append(features[:80, :])  # Toma solo los primeros 80 frames
                labels.append(label)

data = np.array(data)
labels = np.array(labels)

data = np.expand_dims(data, axis=-1)

data = np.expand_dims(data, axis=1)

labels = to_categorical(labels, num_classes=3)

print(f'Data shape: {data.shape}')
print(f'Labels shape: {labels.shape}')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


num_samples = data.shape[0]
zeta = data.shape[1]
num_frames = data.shape[2]
num_features = data.shape[3]
num_speakers = 3

input_shape = (zeta, num_frames, num_features, 1)

inputs = Input(shape=input_shape)

x = Conv3D(filters=36, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(inputs)
x = Conv3D(filters=36, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

x = Conv3D(filters=72, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = Conv3D(filters=72, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

x = Conv3D(filters=144, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = Conv3D(filters=144, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)


x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_speakers, activation='softmax')(x)


model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


model.fit(data, labels, epochs=50, batch_size=4, validation_split=0.2)
model.save('../models/speaker_verification_3D.keras')

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy}')
