import os
import asyncio
import multiprocessing
import time
import tempfile
from moviepy.editor import VideoFileClip
import shutil
from typing import Dict, List
import json
from concurrent.futures import ProcessPoolExecutor
from image import generate_single_image,generate_images_parallel
from narration import generate_audio
from video import create_videos, concatenate_videos_with_transitions, optimize_worker_count
from faster_whisper import WhisperModel
from caption import WhisperWord, process_video_with_emoji_overlays, split_into_natural_phrases
from music import add_background_music
from transcribe_segments import transcribe
from typing import List
import logging
from openai import OpenAI
from status_manager import video_status
import random
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
client = OpenAI(api_key=API_KEY)

async def generate_audio_file(narration: str) -> str:
    """Generate audio file from narration text."""
    video_status.update_step("Generating Audio", "Creating voice narration")
    return await asyncio.to_thread(generate_audio, narration, 'en')
def generate_transitions(count):
    base_transitions = [
        "zoom blend",
        "rotate fade",
        "zoom wipe",
        "reveal zoom",
        "spin",
        "wipe",
        "fade",
        "circle reveal"
    ]
    
    # Calculate how many times we need to repeat the full list to meet the count
    cycles = (count + len(base_transitions) - 1) // len(base_transitions)
    transitions = base_transitions * cycles
    
    # Shuffle the list randomly
    random.shuffle(transitions)
    
    # Return the required number of transitions
    return transitions[:count]

def read_story_segments(file_path: str) -> tuple[List[str], str]:
    """
    Read story segments and full story.
    Returns tuple of (segments list, full story text)
    """
    segments = []
    full_story = ""
    current_segment = ""
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Extract full story by removing segment markers
    full_story = ' '.join(line.strip() for line in lines 
                         if not line.startswith('Segment') and line.strip())
    
    # Extract individual segments
    for line in lines:
        if line.startswith('Segment'):
            if current_segment:  # Save previous segment
                segments.append(current_segment.strip())
            current_segment = ""
        elif line.strip():  # Add non-empty lines to current segment
            current_segment += line.strip() + " "
    
    # Add last segment
    if current_segment:
        segments.append(current_segment.strip())
        
    return segments, full_story
async def regenerate_video(video_id: str, metadata: Dict) -> Dict:
    """Regenerate video with updated transitions and styles"""
    try:
        # Get video segments
        video_segments = []
        for i, segment in enumerate(metadata['segments']):
            # Check if prompt was changed (compare with old_prompt)
            curr_prompt = segment.get('prompt', '')
            old_prompt = segment.get('old_prompt', '')
            
            logger.info(f"Segment {i}: Current prompt = '{curr_prompt}', Old prompt = '{old_prompt}'")
            
            if curr_prompt != old_prompt:
                logger.info(f"Regenerating image for segment {i} with new prompt: {curr_prompt}")
                # Generate new image
                new_image = await generate_single_image(curr_prompt, i + 1, video_id)
                if not new_image:
                    raise Exception(f"Failed to generate new image for segment {i}")
                
                # Create new video segment
                temp_video_path = os.path.join("videos", f"segment_{video_id}_{i}.mp4")
                await asyncio.to_thread(
                    create_videos,
                    [new_image],
                    [""],
                    max_workers=1,
                    output_path=temp_video_path
                )
                
                # Update metadata
                segment['image_path'] = new_image
                
            # Get existing segment path
            vid_path = os.path.join("videos", f"segment_{video_id}_{i}.mp4")
            if os.path.exists(vid_path):
                video_segments.append(vid_path)
            else:
                raise Exception(f"Video segment {i} not found: {vid_path}")

        transitions = [seg.get('transition', 'fade') for seg in metadata['segments'][:-1]]
        
        # Create new uncaptioned video
        new_uncaptioned = os.path.join("videos", f"new_uncaptioned_{video_id}.mp4")
        
        # Get audio file path
        audio_file = os.path.join("audio", f"audio_{video_id}.mp3")
        if not os.path.exists(audio_file):
            logger.info("Audio file not found, regenerating...")
            story_text = " ".join(seg['text'] for seg in metadata['segments'])
            audio_file = await generate_audio_file(story_text)

        await asyncio.to_thread(
            concatenate_videos_with_transitions,
            video_segments,
            audio_file,
            new_uncaptioned,
            transitions=transitions
        )
        
        # Apply captions with styles
        captioned_video = os.path.join("generated_videos", f"video_{video_id}.mp4")
        
        # Initialize Whisper model
        model = WhisperModel(
            model_size_or_path="large-v2",
            device="cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
            compute_type="float16" if os.environ.get('CUDA_VISIBLE_DEVICES') else "int8"
        )
        
        # Get word timestamps
        segments, info = await asyncio.to_thread(
            model.transcribe,
            new_uncaptioned,
            word_timestamps=True
        )
        
        # Process words
        all_words = []
        for segment in segments:
            if hasattr(segment, 'words'):
                all_words.extend([
                    WhisperWord(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        probability=word.probability
                    ) for word in segment.words
                ])
        
        # Split into phrases
        phrases = await asyncio.to_thread(
            split_into_natural_phrases,
            all_words,
            new_uncaptioned
        )
        
        # Apply captions
        await asyncio.to_thread(
            process_video_with_emoji_overlays,
            new_uncaptioned,
            captioned_video,
            phrases,
            caption_style=metadata.get('caption_style', 'default')
        )
        
        # Cleanup
        if os.path.exists(new_uncaptioned):
            os.remove(new_uncaptioned)
            
        # Save updated metadata with new prompts
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error regenerating video: {str(e)}")
        return {"status": "error", "error": str(e)}
async def generate_single_prompt(
    segment: str, 
    full_story: str, 
    segment_index: int, 
    total_segments: int, 
    previous_prompts: List[str] = [], 
    image_style: str = ""
) -> str:
    """Generate image prompt for a single story segment with context."""
    try:
        system_prompt = """You are an expert at crafting focused, precise image prompts. Your task is to generate a single, detailed image prompt that zeros in on the core subject of each story segment.

Key Requirements:

1. FOCUS ON THE MAIN SUBJECT:
- If discussing an engine: Show the detailed engine itself
Example: "A gleaming V8 engine with exposed chrome headers" NOT "A car in a garage"
- Describe the main subject, event, action in the segment.
- Look at the whole narration and refers to it.
Example: if the story is talking about cars, and then it talked about engine. Don't use a image that depicts engine in a motorcycle, create a image prompt that discuss this engine in the car

2. CHARACTER CONSISTENCY:
- When a character appears in multiple segments, maintain exact same appearance throughout
- Copy specific details from previous prompts: clothing, hair, facial features, etc.
- Only introduce new character details if explicitly mentioned in the current segment
- Be very specific about the appearance of the character so that it can be copied in the next video
Example: If previous segment described "a woman with wavy red hair wearing a blue blazer with a sharp face and many pimples", keep these exact details in the next segment

3. ENVIRONMENT AS CONTEXT (MINIMAL):
- Only include environmental elements that directly enhance the main subject
- Keep background simple and relevant
- Keep the background style consistent with the rest of the elements

4. MAINTAIN STORY CONTINUITY:
- Reference previous visuals only if directly relevant to current subject
- Keep consistent lighting and atmosphere across segments
- Ensure visual style matches throughout

Constraints:
- Think creativly to depict the information to the viewer
- Be accurate about the description 
- No text or numbers in the image
- Maximum 2-3 sentences per description
- Only mention characters if they are directly involved in the segment's action
- Make sure it fills out the 9:16 space
Output Format:
Single paragraph with clear description focusing on the main subject. For segments with returning characters, copy their exact appearance from previous descriptions.
"""

        user_prompt = f"""Full story context:
{full_story}

Current segment ({segment_index} of {total_segments}):
{segment}"""

        if previous_prompts:
            user_prompt += "\n\nPrevious image prompts for context reference:"
            for i, prev_prompt in enumerate(previous_prompts, 1):
                user_prompt += f"\nPrompt {i}: {prev_prompt}"
        
        user_prompt += "\n\nGenerate a single image prompt for this segment that fits within the overall story sequence but still expressing the core idea of this segment. Use the rest of the story and image prompts before as reference"
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        prompt = response.choices[0].message.content.strip()
        
        # Append the image style if provided and it's not the default
        if image_style:
            prompt += f" {image_style}"
            
        return prompt
        
    except Exception as e:
        logger.error(f"OpenAI API Error for segment {segment_index}: {str(e)}")
        raise Exception(f"Failed to generate image prompt: {str(e)}")
async def main(story_input: str = None, caption_style: str = "karaoke", image_style: str = "", task_id: str = None) -> dict:
    """Main function that creates a video from text input with improved segment handling."""
    try:
        if not story_input:
            raise ValueError("No story input provided")

        # Initialize timing and status
        video_status.start_time = time.time()
        video_status.status = "processing"
        
        # Create directories with proper error handling
        directories = ["images", "videos", "audio", "emojis", "generated_videos"]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise Exception(f"Failed to create directory {directory}: {str(e)}")

        video_status.progress = 5
        video_status.update_step("Initializing", "Setting up work environment")

        # Transcribe input
        video_status.update_step("Processing Text", "Analyzing and segmenting story")
        await asyncio.to_thread(transcribe, story_input)
        video_status.progress = 15

        # Read segments and generate prompts with validation
        segments, full_story = read_story_segments("story_segment.txt")
        if not segments:
            raise ValueError("No valid story segments found")
            
        total_segments = len(segments)
        video_status.update_step("Creating Prompts", f"0/{total_segments} complete")

        # Generate prompts and initialize metadata
        prompts = []
        metadata = {
            'segments': [],
            'caption_style': caption_style,
            'image_style': image_style,
            'total_segments': total_segments,
            'story': story_input
        }

        # Generate prompts with progress tracking
        for i, segment in enumerate(segments, 1):
            video_status.update_step("Creating Prompts", f"Generating prompt {i}/{total_segments}")
            try:
                prompt = await generate_single_prompt(
                    segment, 
                    full_story, 
                    i, 
                    total_segments,
                    previous_prompts=prompts,
                    image_style=image_style
                )
                prompts.append(prompt)
                
                # Add segment to metadata
                metadata['segments'].append({
                    'index': i-1,
                    'text': segment,
                    'prompt': prompt,
                    'transition': 'fade'  # Default transition
                })
            except Exception as e:
                logger.error(f"Error generating prompt for segment {i}: {str(e)}")
                raise
            
            video_status.progress = 15 + (15 * i // total_segments)

        # Save metadata early for potential error recovery
        metadata_path = f"generated_videos/metadata_{task_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
  
        # Generate images with validation
        video_status.update_step("Generating Images", "Starting image generation")
        images = await generate_images_parallel(prompts, task_id)
        
        if not images:
            raise Exception("Failed to generate any valid images")
        
        if len(images) != total_segments:
            raise Exception(f"Expected {total_segments} images, but generated {len(images)}")

        # Update metadata with verified image paths
        for i, image_path in enumerate(images):
            if not os.path.exists(image_path):
                raise Exception(f"Generated image not found: {image_path}")
            metadata['segments'][i]['image_path'] = image_path
            
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        video_status.progress = 45
        video_status.update_step("Processing Audio", "Generating narration")

        # Generate audio with validation
        audio_file = await generate_audio_file(story_input)
        if not os.path.exists(audio_file):
            raise Exception("Failed to generate audio file")
            
        metadata['audio_path'] = audio_file
        video_status.progress = 55

        # Create and verify individual video segments
        video_status.status = "creating_videos"
        video_files = []
        
        for i, image in enumerate(images, 1):
            video_status.update_step("Creating Videos", f"Processing video segment {i}/{total_segments}")
            
            # Generate segment path
            segment_path = os.path.join("videos", f"segment_{task_id}_{i-1}.mp4")
            
            try:
                # Create individual segment
                await asyncio.to_thread(
                    create_videos,
                    [image],
                    [""],  # No effect prompts
                    max_workers=1,
                    output_path=segment_path
                )
                
                # Verify segment was created and is valid
                if not os.path.exists(segment_path):
                    raise Exception(f"Failed to create video segment {i}")
                
                # Verify video file is valid using VideoFileClip
                with VideoFileClip(segment_path) as clip:
                    if clip.duration <= 0:
                        raise Exception(f"Invalid duration for video segment {i}")
                
                video_files.append(segment_path)
                metadata['segments'][i-1]['video_path'] = segment_path
                
            except Exception as e:
                logger.error(f"Error creating video segment {i}: {str(e)}")
                raise
                
            video_status.progress = 55 + (15 * i // total_segments)

        # Verify we have all required segments
        if len(video_files) != total_segments:
            raise Exception(f"Expected {total_segments} video segments, but created {len(video_files)}")

        # Save metadata with segment information
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Generate transitions sequence
        transition_sequence = generate_transitions(len(video_files) - 1)
        
        # Update metadata with transitions
        for i, transition in enumerate(transition_sequence):
            metadata['segments'][i]['transition'] = transition
            
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Combine videos with transitions
        video_status.update_step("Combining Videos", "Merging segments with transitions")
        uncaptioned_path = os.path.join("videos", f"uncaptioned_{task_id}.mp4")
        
        try:
            await asyncio.to_thread(
                concatenate_videos_with_transitions,
                video_files,
                audio_file,
                uncaptioned_path,
                transitions=transition_sequence
            )
            
            if not os.path.exists(uncaptioned_path):
                raise Exception("Failed to create uncaptioned video")
                
        except Exception as e:
            logger.error(f"Error combining videos: {str(e)}")
            raise

        metadata['uncaptioned_path'] = uncaptioned_path


        video_status.update_step("Processing Captions", "Analyzing speech patterns")
        
        # Initialize Whisper model without time constraints
        model = WhisperModel(
            model_size_or_path="large-v2",
            device="cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
            compute_type="float16" if os.environ.get('CUDA_VISIBLE_DEVICES') else "int8",
            device_index=0,
            cpu_threads=8
        )

        # Do transcription without async/timeout
        segments, info = model.transcribe(
            uncaptioned_path,
            word_timestamps=True,
            language="en",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=100,
                threshold=0.3
            )
        )

        # Process words without timing constraints
        all_words = []
        for segment in segments:
            if hasattr(segment, 'words'):
                all_words.extend([
                    WhisperWord(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        probability=word.probability
                    ) for word in segment.words
                ])

        # Split phrases without timing constraints
        phrases = split_into_natural_phrases(all_words, uncaptioned_path)

        # Process video with captions
        final_video_path = os.path.join("generated_videos", f"video_{task_id}.mp4")
        
        # Remove async to match test.py
        process_video_with_emoji_overlays(
            uncaptioned_path,
            final_video_path,
            phrases,
            caption_style=caption_style
        )

        return {
            "status": "success",
            "video_path": final_video_path,
            "metadata_path": metadata_path
        }

    except Exception as e:
        logger.error(f"Error in caption processing: {str(e)}")
        raise

   