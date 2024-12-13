import captacity
from narration import generate_audio
import math

def calculate_required_segments(duration, segment_duration=3.0):
    """Calculate the exact number of segments needed based on audio duration."""
    return math.ceil(duration / segment_duration)

def group_words_into_segments(start_times, end_times, words, segment_duration=3.0):
    """Group words into segments targeting exactly 3 seconds each, returning segments with indices."""
    if not words:
        return []
        
    segments = []  # Will store tuples of (text, start_idx, end_idx)
    
    # Get total duration and number of full segments
    total_duration = max(end_times)
    num_full_segments = math.floor(total_duration / segment_duration)
    
    current_word_idx = 0
    
    # Handle full 3-second segments
    for segment_num in range(num_full_segments):
        target_end_time = (segment_num + 1) * segment_duration
        current_segment = []
        segment_start_idx = current_word_idx
        
        while current_word_idx < len(words):
            current_segment.append(words[current_word_idx])
            
            # Check if adding the next word would exceed the target time
            if (current_word_idx + 1 < len(words) and 
                end_times[current_word_idx + 1] > target_end_time):
                
                # Determine whether to include the next word based on which boundary is closer
                next_word_boundary = end_times[current_word_idx + 1]
                current_boundary = end_times[current_word_idx]
                
                if (abs(next_word_boundary - target_end_time) < 
                    abs(current_boundary - target_end_time) and 
                    next_word_boundary - start_times[segment_start_idx] <= segment_duration * 1.2):
                    # Include next word if it makes the segment more precise
                    current_word_idx += 1
                    current_segment.append(words[current_word_idx])
                break
                
            current_word_idx += 1
            
            if current_word_idx >= len(words):
                break
        
        if current_segment:
            segments.append((
                ' '.join(current_segment),
                segment_start_idx,
                current_word_idx
            ))
        current_word_idx += 1
    
    # Handle remaining words as the last segment
    if current_word_idx < len(words):
        remaining_words = words[current_word_idx:]
        if remaining_words:
            segments.append((
                ' '.join(remaining_words),
                current_word_idx,
                len(words) - 1
            ))
    
    return segments

def transcribe(narration):
    """Transcribe narration into precisely timed 2.5-second segments."""
    # Generate audio file
    audio_path = generate_audio(narration)
    
    # Get segments from captacity
    segments = captacity.transcriber.transcribe_locally(audio_path)
    captions = captacity.segment_parser.parse(
        segments=segments,
        fit_function=lambda text: len(text.split()) <= 1
    )
    
    # Extract timing information
    start_times = []
    end_times = []
    words = []
    for segment in captions:
        start_times.append(segment['start'])
        end_times.append(segment['end'])
        words.append(segment['text'])
    
    # Get total duration
    total_duration = max(end_times)
    
    # Get segments with 3-second timing and indices
    time_segments = group_words_into_segments(
        start_times, 
        end_times, 
        words,
        segment_duration=2.5
    )
    
    # Write segments to file with timing information
    with open("story_segment.txt", "w") as file:
        for i, (segment_text, start_idx, end_idx) in enumerate(time_segments, 1):
            if segment_text:
                segment_start = start_times[start_idx]
                segment_end = end_times[end_idx]
                file.write(f"Segment {i}:\n")
                file.write(f"{segment_text}\n\n")
