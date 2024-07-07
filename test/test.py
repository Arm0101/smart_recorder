from load import load_verification_model
from speaker_verification import speaker_verification
import os
model = load_verification_model('../models/speaker_identification.keras')
label_map_file = '../models/speaker_identification_label_map.json'
print('model loaded')
test_path = '../temp'
for audio_path in os.listdir(test_path):
    if not audio_path.endswith('.wav'):
        continue
    full_path = os.path.join(test_path, audio_path)
    prediction, predicted_label, predicted_person = speaker_verification(full_path, model, label_map_file)
    print(f'prediction {prediction}\n')
    print(f'{predicted_label}: {predicted_person}')
    print('----------------\n')

