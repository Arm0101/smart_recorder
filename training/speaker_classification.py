import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from training.dataset import prepare_dataset
from utils import save_label_map

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

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

model_save_path = '../models/speaker_identification.keras'
model.save(model_save_path)
