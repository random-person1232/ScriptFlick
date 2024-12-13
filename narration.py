import os
import librosa
import soundfile as sf
from gtts import gTTS

def get_audio_duration(audio_file):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

def generate_audio(narration, language='en'):
    output_dir = "audio"
    os.makedirs(output_dir, exist_ok=True)
    final_audio_file = os.path.join(output_dir, "narration.mp3")

    try:
        # Create gTTS object
        tts = gTTS(text=narration, lang=language, slow=False)
        
        # Save the audio file
        tts.save(final_audio_file)
        
        return final_audio_file
    
    except Exception as e:
        print(f"An error occurred while generating audio: {str(e)}")
        raise