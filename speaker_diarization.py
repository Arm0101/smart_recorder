from pathlib import Path
from pyannote.audio import Pipeline
import os
from pydub import AudioSegment


def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")

    cwd = Path.cwd().resolve()
    cd_to = path_to_config.parent.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    pipeline = Pipeline.from_pretrained(path_to_config)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline


def segment_audio(audio_path, pipeline: Pipeline):
    diarization = pipeline(audio_path)

    audio = AudioSegment.from_wav(audio_path)

    for i, turn in enumerate(diarization.itertracks(yield_label=True)):
        start_time = turn[0].start * 1000
        end_time = turn[0].end * 1000
        speaker_label = turn[2]
        segment = audio[start_time:end_time]
        segment.export(f"speaker_{speaker_label}_segment_{i}.wav", format="wav")


PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"
pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)
