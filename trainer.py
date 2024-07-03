from dataset import prepare_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback

path = 'records'

model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name,
                                                          num_labels=len(set(prepare_dataset('records')['label'])))

print('model loaded')


def preprocess_function(examples):
    audio = examples["path"]
    inputs = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    inputs['labels'] = examples['label']
    return inputs


dataset = prepare_dataset(path)
encoded_dataset = dataset.map(preprocess_function, remove_columns=["path"])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=feature_extractor,
    callbacks=[TensorBoardCallback()]
)

print("start training...")
trainer.train()

model.save_pretrained("./fine_tuned_speaker_identification")
feature_extractor.save_pretrained("./fine_tuned_speaker_identification")
