import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import ogg_to_wav
from dataset import prepare_dataset,extract_features, convert_audios_to_wav


lstm_units = 128
dense_units = 64
dropout_rate = 0.4
learning_rate = 0.0005


# Path to the records directory
base_path = 'records'
data, labels, label_map = prepare_dataset(base_path)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Validation size: {X_val.shape}")
print(f"Test size: {X_test.shape}")

# Definir la arquitectura de la red neuronal con hiperparámetros ajustados
model = Sequential()
model.add(tf.keras.layers.Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(lstm_units))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(len(label_map), activation='softmax'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Reajustar los datos para que sean compatibles con la entrada de la LSTM
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Guardar el modelo en formato .keras
model_save_path = './fine_tuned_speaker_identification.keras'
model.save(model_save_path)



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
# audio_path = 'records/ovidio/audio_2024-07-03_11-56-37.wav'


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