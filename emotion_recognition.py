import os
import pandas as pd
import librosa
from datasets import Dataset

# Define la ruta al directorio de audio CREMA-D
data_dir = "/content/cremad/AudioWAV"

# Crear un dataframe para almacenar las rutas de los archivos y las etiquetas
data = []

# Mapear las emociones a números
emotion_map = {
    "SAD": 0,
    "ANG": 1,
    "DIS": 2,
    "FEA": 3,
    "HAP": 4,
    "NEU": 5
}

# Recorrer los archivos de audio y extraer las etiquetas
for file in os.listdir(data_dir):
    if file.endswith(".wav"):
        emotion = file.split('_')[2]  # Extraer la emoción del nombre del archivo
        if emotion in emotion_map:
            label = emotion_map[emotion]
            file_path = os.path.join(data_dir, file)
            data.append({"file_path": file_path, "label": label})


# Convertir a un dataframe
df = pd.DataFrame(data)

# Convertir a Hugging Face Dataset
dataset = Dataset.from_pandas(df)


from transformers import Wav2Vec2Processor

# Asegúrate de tener el procesador correcto
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Función para procesar los audios
def preprocess_function(examples):
    audio = examples["file_path"]
    speech_array, sampling_rate = librosa.load(audio, sr=16000)
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    examples["input_values"] = inputs.input_values[0]
    examples["labels"] = examples["label"]
    return examples

print(dataset)
# Aplica la preparación del conjunto de datos
dataset = dataset.map(preprocess_function, remove_columns=["file_path", "label"])


# Divide el conjunto de datos si es necesario
dataset = dataset.train_test_split(test_size=0.1)


# Define los argumentos de entrenamiento
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# Define el modelo
from transformers import Wav2Vec2ForSequenceClassification

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(emotion_map),
)

# Define el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

# Entrena el modelo
trainer.train()

# Guarda el modelo y el procesador
model.save_pretrained("models/emo_model")
processor.save_pretrained("models/emo_model")



