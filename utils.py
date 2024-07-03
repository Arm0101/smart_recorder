from pydub import AudioSegment


def ogg_to_wav(ogg_file, wav_file):
    audio = AudioSegment.from_ogg(ogg_file)
    audio.export(wav_file, format="wav")
