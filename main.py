import os
from load import load_verification_model, load_emo_model
from utils import record_audio
from speaker_diarization import load_pipeline_from_pretrained
from matching_networks import speaker_verification, add_new_classes
from speaker_diarization import segment_audio
from emotion_recognition import predict_emotion
from transcription import transcribe_audio
import librosa
import sounddevice as sd
import shutil
import tensorflow as tf
from utils import load_label_map
if __name__ == '__main__':
    temp_path = 'temp'
    audio_name = 'audio.wav'
    label_map_file = 'models/speaker_identification_label_map.json'

    PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"
    pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)
    classification_model = load_verification_model('models/speaker_identification.keras')
    emo_model, emo_model_processor = load_emo_model('models/emo_model')
    embedding_model = tf.keras.models.load_model('models/speaker_identification_embedding.keras')
    support_set_file_path = 'models/speaker_support_set_embeddings.pkl'

    s, labels, oc_svm = add_new_classes('new_dataset/', embedding_model, support_set_file_path)
    label_map = load_label_map(label_map_file)
    print('models loaded')

    # List available audio devices
    print(sd.query_devices())
    device = int(input('select device: '))

    while True:
        duration = input('Duration: ')
        try:
            duration = int(duration)
        except ValueError:
            break
        if (os.path.
                exists(temp_path)):
            shutil.rmtree(temp_path)
            os.makedirs(temp_path)

        record_audio(duration=duration, output_dir=temp_path, output_filename=audio_name, device=device)
        _audio_path = os.path.join(temp_path, audio_name)
        if not os.path.exists(_audio_path):
            exit(1)

        segment_audio(audio_path=_audio_path, pipeline=pipeline)

        temp_files = filter(lambda f: f != audio_name and f.endswith('.wav'), os.listdir(temp_path))

        for audio_path in temp_files:

            audio, sr = librosa.load(os.path.join(temp_path, audio_path))
            if librosa.get_duration(y=audio, sr=sr) < 1:
                continue

            full_path = person_path = os.path.join(temp_path, audio_path)

            prediction, _ = speaker_verification(full_path, embedding_model, support_set_file_path, oc_svm, label_map)
            print(f'prediction {prediction}\n')
            print(f'{prediction}')

            predicted_emotion = predict_emotion(full_path, emo_model, emo_model_processor)
            print(f"emotion: {predicted_emotion}")

            text = transcribe_audio(full_path)
            print(f'transcription: {text}')

