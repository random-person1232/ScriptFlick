import os
import pyttsx3
import librosa
import soundfile as sf

def get_audio_duration(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)
    
    # Calculate the duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return duration

import os
import pyttsx3
import librosa
import soundfile as sf

def get_audio_duration(audio_file):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

def generate_audio(narration, language='en'):
    output_dir = "audio"
    final_audio_file = os.path.join(output_dir, "narration.mp3")

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if language in voice.languages:
                engine.setProperty('voice', voice.id)
                break
        
        engine.save_to_file(narration, final_audio_file)
        engine.runAndWait()
        
        return final_audio_file
    
    except Exception as e:
        print(f"An error occurred while generating audio: {str(e)}")
        raise