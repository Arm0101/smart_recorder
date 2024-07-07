import os
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Reshape, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def load_data(data_path, n_mfcc=40, max_len=500):
    labels = []
    mfccs = []
    speakers = ['jose', 'armando', 'ovidio']
    for i, speaker in enumerate(speakers):
        speaker_path = os.path.join(data_path, speaker)
        for audio_file in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, audio_file)
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            if mfcc.shape[1] < max_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]
            mfccs.append(mfcc)
            labels.append(i)
    mfccs = np.array(mfccs)
    labels = to_categorical(labels)
    return mfccs, labels, speakers


data_path = '../dataset2'
mfccs, labels, speakers = load_data(data_path)

X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]  # Agregar dimensión de canal
X_train = np.expand_dims(X_train, axis=1)  # Agregar dimensión de profundidad (zeta)
X_test = X_test[..., np.newaxis]  # Agregar dimensión de canal
X_test = np.expand_dims(X_test, axis=1)  # Agregar dimensión de profundidad (zeta)

input_shape = (1, 40, 500, 1)  # Ajustado para audios de hasta 5 segundos con MFCCs

inputs = Input(shape=input_shape)

# 3D CNN layers con Dropout
x = Conv3D(filters=36, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(inputs)
x = Dropout(0.3)(x)
x = Conv3D(filters=36, kernel_size=(1, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(x)
x = Conv3D(filters=72, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), padding='valid')(x)
x = Dropout(0.3)(x)
x = Conv3D(filters=72, kernel_size=(1, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(x)
x = Conv3D(filters=144, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), padding='valid')(x)
x = Dropout(0.3)(x)

shape_before_reshape = list(x.shape)
print(f"Shape before Reshape: {shape_before_reshape}")

_, depth, height, width, channels = shape_before_reshape
new_shape = (height, width * channels, depth)

# Reshape and 2D CNN layers
x = Reshape(new_shape)(x)

x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Dropout(0.3)(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Dropout(0.3)(x)
x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(speakers), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks para Early Stopping y reducción de la tasa de aprendizaje
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

batch_size = 32
epochs = 50

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr])

model.save('../models/speaker_verification_hyb_cnn.keras')
