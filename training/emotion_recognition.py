# google collab
import os
import pandas as pd
import librosa
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForSequenceClassification

data_dir = "/content/cremad/AudioWAV"

data = []

emotion_map = {
    "SAD": 0,
    "ANG": 1,
    "DIS": 2,
    "FEA": 3,
    "HAP": 4,
    "NEU": 5
}

for file in os.listdir(data_dir):
    if file.endswith(".wav"):
        emotion = file.split('_')[2]  # Extraer la emoción del nombre del archivo
        if emotion in emotion_map:
            label = emotion_map[emotion]
            file_path = os.path.join(data_dir, file)
            data.append({"file_path": file_path, "label": label})

df = pd.DataFrame(data)

dataset = Dataset.from_pandas(df)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


def preprocess_function(examples):
    audio = examples["file_path"]
    speech_array, sampling_rate = librosa.load(audio, sr=16000)
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    examples["input_values"] = inputs.input_values[0]
    examples["labels"] = examples["label"]
    return examples


dataset = dataset.map(preprocess_function, remove_columns=["file_path", "label"])


dataset = dataset.train_test_split(test_size=0.1)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)


model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(emotion_map),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()

model_save_path = '../models/emo_model'
model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)
