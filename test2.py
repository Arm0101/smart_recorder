import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import ogg_to_wav
from dataset import convert_audios_to_wav, prepare_dataset, extract_features
from tensorflow.keras.callbacks import EarlyStopping




# Ajuste de hiperparámetros
conv_filters = 64
conv_kernel_size = 3
conv_stride = 1
conv_activation = 'relu'
pool_size = 2

lstm_units = 128
dense_units = 64
dropout_rate = 0.4
learning_rate = 0.0005

# Ruta al directorio de los registros
base_path = 'records'
convert_audios_to_wav(base_path)
data, labels, label_map = prepare_dataset(base_path)
print(label_map)

# Dividir el conjunto de datos en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Validation size: {X_val.shape}")
print(f"Test size: {X_test.shape}")

# Reajustar los datos para que sean compatibles con la entrada de Conv1D y LSTM
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Definir la nueva arquitectura de la red neuronal
model = Sequential()
model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_stride, activation=conv_activation,
                 input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Dropout(dropout_rate))

model.add(LSTM(lstm_units, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(lstm_units))
model.add(Dropout(dropout_rate))

model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(len(label_map), activation='softmax'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo con early stopping
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Guardar el modelo en formato .keras
model_save_path = './fine_tuned_speaker_identification.keras'
model.save(model_save_path)

# Verificar si el archivo existe antes de cargarlo
if os.path.exists(model_save_path):
    # Cargar el modelo guardado
    loaded_model = tf.keras.models.load_model(model_save_path)
    print("Modelo cargado exitosamente.")
else:
    print(f"El archivo {model_save_path} no existe. Por favor, verifica la ruta.")

# Ejemplo de cómo preprocesar un nuevo archivo de audio para predicción
input_audio = 'jose01.ogg'
audio_path = 'jose01.wav'
ogg_to_wav(input_audio, audio_path)

features = extract_features(audio_path)

# Reajustar las dimensiones para que coincidan con la entrada del modelo
features = np.expand_dims(features, axis=0)  # Añadir dimensión de lote
features = np.expand_dims(features, axis=-1)  # Añadir dimensión de canal

# Realizar una predicción
prediction = loaded_model.predict(features)
print(f'prediction {prediction}\n')
predicted_label = np.argmax(prediction, axis=1)

# Mapear el índice de la etiqueta a la persona correspondiente
label_map_inverted = {v: k for k, v in label_map.items()}  # Invertir el mapa de etiquetas
predicted_person = label_map_inverted[predicted_label[0]]

print(f"Predicted label: {predicted_label}")
print(f"Predicted person: {predicted_person}")
