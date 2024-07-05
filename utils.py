from pydub import AudioSegment
import os


def ogg_to_wav(ogg_file, wav_file):
    audio = AudioSegment.from_ogg(ogg_file)
    audio.export(wav_file, format="wav")


def convert_audios_to_wav(path):
    for f in os.listdir(path):
        _path = os.path.join(path, f)
        if os.path.isdir(_path):
            for file_name in os.listdir(_path):
                if file_name.endswith('.ogg'):
                    file_path = os.path.join(_path, file_name)
                    output_path = os.path.splitext(file_path)[0] + '.wav'
                    if not os.path.exists(output_path):
                        ogg_to_wav(file_path, output_path)
                    os.remove(file_path)
                    print(file_path)
        elif _path.endswith('.ogg'):
            output_path = os.path.splitext(_path)[0] + '.wav'
            ogg_to_wav(_path, output_path)
            os.remove(_path)


def preprocess_recording(audio_path, output_dir='', n_seg=10):
    if not output_dir:
        output_dir = os.path.dirname(audio_path)

    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio)
    segment_duration = n_seg * 1000

    for i in range(0, audio_duration, segment_duration):
        segment = audio[i:i + segment_duration]
        segment_filename = os.path.join(output_dir,
                                        f"{os.path.basename(audio_path).split('.')[0]}_segment_{i // segment_duration}.wav")
        segment.export(segment_filename, format="wav")
