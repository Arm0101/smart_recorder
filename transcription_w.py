import whisperx


def load_model(path):
    device = "cpu"
    model = whisperx.load_model(path, device, compute_type='int8')
    return model


def transcribe_audio(audio_file, model):
    result = model.transcribe(audio_file, language='es')
    text = ''
    for r in result['segments']:
        text += f' {r['text']}'
    return text


