from pydub import AudioSegment
import sounddevice as sd
import wave
import numpy as np
import os
import json


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
        if len(segment) < 3000:
            continue
        segment_filename = os.path.join(output_dir,
                                        f"{os.path.basename(audio_path).split('.')[0]}_segment_{i // segment_duration}.wav")
        segment.export(segment_filename, format="wav")


def rename_w_audio(path):
    for file in os.listdir(path):
        file_name = file.split(' ')[2:]
        file_name = '_'.join(file_name)
        name, ext = os.path.splitext(file_name)
        file_name = name.replace('.', '-') + ext
        file_name = os.path.join(path, file_name)
        os.rename(os.path.join(path, file), file_name)


def save_label_map(label_map, filename):
    with open(filename, 'w') as f:
        json.dump(label_map, f)


def load_label_map(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def record_audio(duration, output_dir='temp', output_filename='audio.wav', device=None):
    # Parameters
    sample_rate = 44100  # Sample rate
    channels = 1  # Number of audio channels
    dtype = np.int16  # Data type for recording

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Full path for the output file
    output_file_path = os.path.join(output_dir, output_filename)

    print(f"Recording for {duration} seconds...")

    # Frames list to store the recorded data
    frames = []

    # Callback function to capture audio
    def callback(indata, _frames, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype, device=device, callback=callback):
        sd.sleep(int(duration * 1000))  # Convert duration to milliseconds and sleep

    print("Recording finished.")

    # Convert frames to a numpy array
    audio_data = np.concatenate(frames, axis=0)

    # Save the recorded data as a WAV file
    with wave.open(output_file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(np.dtype(dtype).itemsize)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
