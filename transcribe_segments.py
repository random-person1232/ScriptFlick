import captacity
import pyttsx3
import math

def audio(narration, language='en'):
    final_audio_file = "audio_output.mp3"
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

def calculate_required_segments(duration, segment_duration=3.0):
    """Calculate the exact number of segments needed based on audio duration."""
    return math.ceil(duration / segment_duration)

def group_words_into_segments(start_times, end_times, words, segment_duration=3.0):
    """Group words into segments based on strict time boundaries."""
    max_time = max(end_times)
    
    # Calculate exact number of segments needed based on audio duration
    num_segments = calculate_required_segments(max_time, segment_duration)
    
    # Initialize segments list
    segments = [[] for _ in range(num_segments)]
    
    # Calculate segment boundaries
    boundaries = [i * segment_duration for i in range(num_segments + 1)]
    
    # Assign words to segments based on their start times
    for word, start_time, end_time in zip(words, start_times, end_times):
        # Find which segment this word belongs to based on start time
        for i, (boundary_start, boundary_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if boundary_start <= start_time < boundary_end:
                segments[i].append(word)
                break
            elif i == len(segments) - 1 and start_time >= boundary_start:
                # Handle words that belong to the last segment
                segments[i].append(word)
    
    # Join words in each segment
    segment_texts = [' '.join(segment) if segment else '' for segment in segments]
    
    return segment_texts

def transcribe(narration):
    
    
    audio_path = audio(narration)
    segments = captacity.transcriber.transcribe_locally(audio_path)
    captions = captacity.segment_parser.parse(
        segments=segments,
        fit_function=lambda text: len(text.split()) <= 1
    )
    
    start_times = []
    end_times = []
    words = []
    for segment in captions:
        start_times.append(segment['start'])
        end_times.append(segment['end'])
        words.append(segment['text'])
    
    # Get total audio duration
    total_duration = max(end_times)
    
    # Get segments
    time_segments = group_words_into_segments(
        start_times, 
        end_times, 
        words, 
        segment_duration=3.0
    )
    
    # Print numbered segments with blank lines between them
    with open("story_segment.txt", "w") as file:
        for i, segment in enumerate(time_segments, 1):
            if segment:  # Only write non-empty segments
                file.write(f"Segment {i}:\n")
                file.write(segment + "\n\n")

transcribe("Deep in the universe, stars gleamed like scattered embers in an endless void. But for each beacon of light, countless others lay hidden, veiled by fear. This was the Black Forest Theory: the cosmos as a dark, silent forest. Every civilization was a hunter, armed and wary, knowing that revealing its presence could invite destruction.A curious species once lit its torch, sending signals into the abyss. They hoped for kinship but found annihilation instead. The silence deepened.Now, those who survived crept quietly, unseen but vigilant, knowing the rule of the forest: in the dark, only the silent endure.")
#https://69shu.biz/c/51232/33449234.html