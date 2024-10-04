from transcription_w import transcribe_audio, load_model
import json
import os
import string
import unicodedata
import difflib

transcriptions = json.load(open('../test/transcriptions/transcriptions.json', 'r'))
path = '../test/transcriptions'


def calculate_similarity(text, original_text):
    similarity_ratio = difflib.SequenceMatcher(None, original_text, text).ratio()

    similarity_percentage = similarity_ratio * 100
    return similarity_percentage


def compare(original_text, text):
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    original_text = ''.join(
        c for c in unicodedata.normalize('NFD', original_text)
        if unicodedata.category(c) != 'Mn'
    )

    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    original_text = original_text.translate(str.maketrans('', '', string.punctuation)).lower()
    print('original: ', original_text)
    print('result: ', text)
    return calculate_similarity(original_text, text)


def main():
    model = load_model('../models/whisper_small/')
    similarity_percentage = []
    for file in os.listdir(path):
        audio = os.path.join(path, file)
        if file.endswith('.wav'):
            text = transcriptions.get(file)
            result = transcribe_audio(audio, model)
            similarity = compare(text, result)
            similarity_percentage.append({'file': file, 'similarity': similarity})
            print('-------------------------------')

    with open('transcription_results_w.json', 'w') as json_file:
        json.dump(similarity_percentage, json_file)


main()
