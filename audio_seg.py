from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import os
from huggingface_hub import hf_hub_download

load_dotenv()


def download_model(output_path):
    api_key = os.getenv('HF_KEY')
    filename = "pytorch_model.bin"
    hf_hub_download(repo_id="pyannote/speaker-diarization-3.1", filename=filename, cache_dir=output_path,
                    use_auth_token=api_key)


download_model('models/speaker-diarization-3.1')

# # Ruta al archivo de audio
# audio_path = "audios/audio.wav"
#
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=api_key)
#
# diarization = pipeline(audio_path)
#
# audio = AudioSegment.from_wav(audio_path)
#
# for i, turn in enumerate(diarization.itertracks(yield_label=True)):
#     start_time = turn[0].start * 1000
#     end_time = turn[0].end * 1000
#     speaker_label = turn[2]
#     segment = audio[start_time:end_time]
#     segment.export(f"speaker_{speaker_label}_segment_{i}.wav", format="wav")
#
# print("Completed.")
