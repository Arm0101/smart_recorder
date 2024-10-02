import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from training.dataset import prepare_dataset
from utils import save_label_map
import matplotlib.pyplot as plt
import pickle
from prototypical_networks import calculate_prototypes

conv_filters = 64
conv_kernel_size = 3
conv_stride = 1
conv_activation = 'relu'
pool_size = 2

lstm_units = 128
dense_units = 64
dropout_rate = 0.4
learning_rate = 0.0001

base_path = '../dataset'
cache_file = '../models/speaker_identification.pkl'
data, labels, label_map = prepare_dataset(base_path, cache_file=cache_file, use_cache=cache_file)
save_label_map(label_map, '../models/speaker_identification_label_map.json')

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Validation size: {X_val.shape}")
print(f"Test size: {X_test.shape}")

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Definir el modelo de clasificación de hablantes
model = Sequential()
model.add(tf.keras.Input(shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_stride, activation=conv_activation))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Dropout(dropout_rate))

model.add(LSTM(lstm_units, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(lstm_units))
model.add(Dropout(dropout_rate))

model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(len(label_map), activation='softmax'))

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32)

model_embedding = Sequential(model.layers[:-1])

train_embeddings = model_embedding.predict(X_train)
val_embeddings = model_embedding.predict(X_val)
test_embeddings = model_embedding.predict(X_test)

prototypes = calculate_prototypes(train_embeddings, y_train, label_map)

prototypes_save_path = '../models/speaker_prototypes.pkl'
with open(prototypes_save_path, 'wb') as file:
    pickle.dump(prototypes, file)

support_set_embeddings = model_embedding.predict(X_train)

support_set_save_path = '../models/speaker_support_set_embeddings.pkl'
with open(support_set_save_path, 'wb') as file:
    pickle.dump((support_set_embeddings, y_train), file)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

model_save_path = '../models/speaker_identification.keras'
model.save(model_save_path)

embedding_model_save_path = '../models/speaker_identification_embedding.keras'
model_embedding.save(embedding_model_save_path)


def plot_embeddings(embeddings, labels, label_map):
    tsne = TSNE(n_components=2, random_state=42)  # t-SNE para reducir los embeddings a 2D
    embeddings_2d = tsne.fit_transform(embeddings)

    label_map_inverted = {v: k for k, v in label_map.items()}

    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        idx = labels == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label_map_inverted[i], alpha=0.6)

    plt.title("Visualización de los Embeddings de las Clase")
    plt.legend()
    plt.show()


plot_embeddings(train_embeddings, y_train, label_map)

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'ro-', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo-', label='Precisión de entrenamiento')
plt.plot(epochs, val_accuracy, 'ro-', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()
