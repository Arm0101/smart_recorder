import speech_recognition as sr


def transcribe_audio(audio_file):
    r = sr.Recognizer()
    # use the audio file as the audio source
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # read the entire audio file

        # recognize speech using Google Speech Recognition
        try:
            text = r.recognize_google(audio, language='es-ES')
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

