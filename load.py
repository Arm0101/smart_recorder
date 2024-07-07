from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import tensorflow as tf
import os


def load_verification_model(model_save_path):
    if os.path.exists(model_save_path):
        loaded_model = tf.keras.models.load_model(model_save_path)
        return loaded_model

    return None


def load_emo_model(model_save_path):
    if os.path.exists(model_save_path):
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_save_path)
        processor = Wav2Vec2Processor.from_pretrained(model_save_path)
        return model, processor
    return None

