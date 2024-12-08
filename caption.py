import tempfile
import subprocess
import sys
from typing import List, Dict, TypedDict, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import random
import re
from emoji import EmojiMapper
import os
from PIL import Image, ImageDraw, ImageFont
import sys
from typing import Optional, Tuple
import logging
import requests
import hashlib
import io
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WhisperWord:
    """Type definition for words from Whisper transcription"""
    word: str
    start: float
    end: float
    probability: float

class PhraseDict(TypedDict):
    """Type definition for phrase dictionaries"""
    top_line: List[WhisperWord]
    bottom_line: List[WhisperWord]
    start: float
    end: float

class SystemRequirements:
    REQUIRED_FONTS = [
        "Arial Unicode MS",
        "Segoe UI Emoji",  # Windows fallback
        "Apple Color Emoji",  # macOS fallback
        "Noto Color Emoji",  # Linux fallback
    ]
    
    MIN_FFMPEG_VERSION = (4, 3, 0)
    
    @staticmethod
    def get_font_paths() -> List[str]:
        """Get system font directories based on OS."""
        font_paths = {
            "windows": [
                "C:\\Windows\\Fonts",
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Windows\\Fonts"),
            ],
            "darwin": [  # macOS
                "/System/Library/Fonts",
                "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ],
            "linux": [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                "/usr/share/fonts/truetype",
                "/usr/share/fonts/opentype",
            ]
        }
        
        if sys.platform.startswith("win"):
            return font_paths["windows"]
        elif sys.platform.startswith("darwin"):
            return font_paths["darwin"]
        else:
            return font_paths["linux"]

    @staticmethod
    def check_font_availability() -> Tuple[bool, List[str]]:
        """Check if required fonts are available."""
        available_fonts = []
        font_paths = SystemRequirements.get_font_paths()
        
        def find_font_file(font_name: str) -> str:
            common_extensions = ['.ttf', '.ttc', '.otf']
            for path in font_paths:
                if not os.path.exists(path):
                    continue
                    
                for ext in common_extensions:
                    # Check for exact matches
                    exact_match = os.path.join(path, f"{font_name}{ext}")
                    if os.path.exists(exact_match):
                        return exact_match
                    
                    # Check for case-insensitive matches
                    try:
                        for file in os.listdir(path):
                            if file.lower().startswith(font_name.lower()) and \
                               any(file.lower().endswith(ext.lower()) for ext in common_extensions):
                                return os.path.join(path, file)
                    except Exception as e:
                        logger.warning(f"Error accessing font directory {path}: {e}")
                        continue
            return ""

        for font in SystemRequirements.REQUIRED_FONTS:
            font_path = find_font_file(font)
            if font_path:
                available_fonts.append(font)
                logger.info(f"Found font: {font} at {font_path}")
        
        has_required_fonts = bool(available_fonts)
        return has_required_fonts, available_fonts

    @staticmethod
    def format_font_path_for_ffmpeg(path: str) -> str:
        """Format a font path for FFmpeg command line usage."""
        if sys.platform.startswith("win"):
            # Escape backslashes and colons for Windows paths in FFmpeg
            return path.replace("\\", "/").replace(":", "\\\\:")
        return path

class EmojiImageGenerator:
    def __init__(self, emoji_dir: str = "emojis"):
        """Initialize emoji generator with directory path."""
        self.emoji_dir = emoji_dir
        if not os.path.exists(emoji_dir):
            os.makedirs(emoji_dir)
            
    def get_twemoji_url(self, emoji_char: str) -> str:
        """Convert emoji character to Twemoji URL."""
        # Get the unicode code points of the emoji character
        code_points = '-'.join(
            hex(ord(char))[2:].lower() 
            for char in emoji_char 
            if char != '\ufe0f'
        )
        return f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{code_points}.png"

    def create_emoji_image(self, emoji_char: str, size: int = 256) -> Optional[str]:
        """Create a PNG image from an emoji character using Twemoji."""
        try:
            # Create unique filename based on emoji character
            emoji_hash = hashlib.md5(emoji_char.encode()).hexdigest()
            filename = f"emoji_{emoji_hash}.png"
            output_path = os.path.join(self.emoji_dir, filename)
            
            # Return existing file if already created
            if os.path.exists(output_path):
                logger.debug(f"Using existing emoji file: {output_path}")
                return output_path
            
            # Try to download from Twemoji first
            try:
                url = self.get_twemoji_url(emoji_char)
                response = requests.get(url)
                response.raise_for_status()
                
                # Load and resize the image
                img = Image.open(io.BytesIO(response.content))
                img = img.convert('RGBA')
                
                # Resize if needed
                if size != img.size[0]:
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                
                # Save the image
                img.save(output_path, 'PNG')
                logger.info(f"Created emoji image from Twemoji: {output_path}")
                return output_path
                
            except Exception as twemoji_error:
                logger.warning(f"Failed to download from Twemoji: {twemoji_error}")
                
                # Fallback to system font rendering
                try:
                    # Create larger initial image for high quality
                    canvas_size = size * 4
                    img = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    
                    # Try to get system emoji font
                    font_size = canvas_size // 2
                    if sys.platform.startswith('win'):
                        font = ImageFont.truetype('seguiemj.ttf', font_size)
                    elif sys.platform.startswith('darwin'):
                        font = ImageFont.truetype('Apple Color Emoji.ttc', font_size)
                    else:
                        font = ImageFont.truetype('NotoColorEmoji.ttf', font_size)
                    
                    # Calculate text position
                    bbox = draw.textbbox((0, 0), emoji_char, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    x = (canvas_size - text_width) // 2
                    y = (canvas_size - text_height) // 2
                    
                    # Draw emoji
                    draw.text((x, y), emoji_char, font=font, embedded_color=True)
                    
                    # Crop to content
                    bbox = img.getbbox()
                    if bbox:
                        # Add padding
                        padding = canvas_size // 8
                        left, top, right, bottom = bbox
                        left = max(0, left - padding)
                        top = max(0, top - padding)
                        right = min(canvas_size, right + padding)
                        bottom = min(canvas_size, bottom + padding)
                        img = img.crop((left, top, right, bottom))
                    
                    # Make square
                    max_size = max(img.size)
                    square = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))
                    paste_x = (max_size - img.width) // 2
                    paste_y = (max_size - img.height) // 2
                    square.paste(img, (paste_x, paste_y), img)
                    
                    # Final resize
                    square = square.resize((size, size), Image.Resampling.LANCZOS)
                    
                    # Save the image
                    square.save(output_path, 'PNG')
                    logger.info(f"Created emoji image using system font: {output_path}")
                    return output_path
                    
                except Exception as font_error:
                    logger.error(f"Failed to create emoji using system font: {font_error}")
                    return None
        
        except Exception as e:
            logger.error(f"Error creating emoji image: {e}")
            return None

def select_most_important_word(words: List[WhisperWord]) -> Optional[WhisperWord]:
    """Select the most important word from a list of words using NLP."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    word_scores = []
    for word in words:
        # Remove punctuation
        cleaned_word = re.sub(r'[^\w\s]', '', word.word)
        # Convert to lowercase
        cleaned_word = cleaned_word.lower()
        # Lemmatize the word
        lemma = lemmatizer.lemmatize(cleaned_word)
        # Check if it's a stop word or empty
        if lemma and lemma not in stop_words:
            # Assign a basic score (could be enhanced with more complex metrics)
            word_scores.append((word, len(lemma)))  # Example: longer words get higher score
    
    if word_scores:
        # Select the word with the highest score
        most_important_word = max(word_scores, key=lambda x: x[1])[0]
        return most_important_word
    else:
        return None

def format_ass_time(seconds: float) -> str:
    """Format time for ASS subtitles."""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        centiseconds = int((secs * 100) % 100)
        secs = int(secs)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    except Exception as e:
        logger.error(f"Error formatting time: {e}")
        return "0:00:00.00"

def calculate_phrase_width(words: List[WhisperWord], font_size: int) -> float:
    """Calculate total width of a phrase."""
    char_width_map = {
        'i': 0.3, 'l': 0.3, 'I': 0.3, '!': 0.3, '.': 0.3, ',': 0.3, "'": 0.3,
        'w': 0.9, 'm': 0.9, 'W': 0.9, 'M': 0.9,
        ' ': 0.3,
    }
    default_width = 0.5
    
    total_width = 0
    for i, word in enumerate(words):
        # Add word width
        word_width = sum(char_width_map.get(c, default_width) * font_size for c in word.word)
        total_width += word_width
        # Add space width if not last word
        if i < len(words) - 1:
            total_width += char_width_map[' '] * font_size
    
    return total_width

def split_into_natural_phrases(words: List[WhisperWord], input_video_path: str) -> List[PhraseDict]:
    """Split words into natural phrases based on timing, punctuation, and max words per phrase."""
    # Get video width directly
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width',
        '-of', 'json',
        input_video_path
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    probe_data = json.loads(probe_result.stdout)
    video_width = int(probe_data['streams'][0]['width'])
    
    phrases: List[PhraseDict] = []
    current_phrase: List[WhisperWord] = []
    phrase_breaks = '.!?,:;'
    MAX_WORDS_PER_PHRASE = 5
    FONT_SIZE = 56
    MAX_WIDTH = video_width * 0.8  # 80% of video width

    def check_split_feasibility(words_to_check: List[WhisperWord]) -> bool:
        """Check if the words can be split into two lines that fit the width."""
        if not words_to_check:
            return True
            
        mid_point = len(words_to_check) // 2
        top_line = words_to_check[:mid_point]
        bottom_line = words_to_check[mid_point:]
        
        top_width = calculate_phrase_width(top_line, FONT_SIZE)
        bottom_width = calculate_phrase_width(bottom_line, FONT_SIZE)
        
        return top_width <= MAX_WIDTH and bottom_width <= MAX_WIDTH

    def find_optimal_split(words_to_check: List[WhisperWord]) -> int:
        """Find optimal number of words that can fit within width constraints."""
        for length in range(len(words_to_check), 0, -1):
            if check_split_feasibility(words_to_check[:length]):
                return length
        return 1

    i = 0
    while i < len(words):
        word = words[i]
        current_phrase.append(word)

        should_break = (
            any(p in word.word for p in phrase_breaks) or
            (i < len(words) - 1 and words[i + 1].start - word.end > 0.25) or
            len(current_phrase) >= MAX_WORDS_PER_PHRASE or
            (len(current_phrase) >= 2 and word.word.lower() in ['and', 'but', 'or', 'because', 'so'])
        )

        if should_break or i == len(words) - 1:
            if current_phrase:
                if not check_split_feasibility(current_phrase):
                    optimal_length = find_optimal_split(current_phrase)
                    phrase_words = current_phrase[:optimal_length]
                    remaining_words = current_phrase[optimal_length:]
                    
                    num_words = len(phrase_words)
                    if num_words <= 2:
                        phrase_obj: PhraseDict = {
                            'top_line': [],
                            'bottom_line': phrase_words,
                            'start': phrase_words[0].start,
                            'end': phrase_words[-1].end + 0.1
                        }
                    else:
                        mid_point = num_words // 2
                        phrase_obj = {
                            'top_line': phrase_words[:mid_point],
                            'bottom_line': phrase_words[mid_point:],
                            'start': phrase_words[0].start,
                            'end': phrase_words[-1].end + 0.1
                        }
                    phrases.append(phrase_obj)
                    
                    current_phrase = remaining_words
                    i = i - (len(remaining_words) - 1)
                else:
                    num_words = len(current_phrase)
                    if num_words <= 2:
                        phrase_obj = {
                            'top_line': [],
                            'bottom_line': current_phrase,
                            'start': current_phrase[0].start,
                            'end': current_phrase[-1].end + 0.1
                        }
                    else:
                        mid_point = num_words // 2
                        phrase_obj = {
                            'top_line': current_phrase[:mid_point],
                            'bottom_line': current_phrase[mid_point:],
                            'start': current_phrase[0].start,
                            'end': current_phrase[-1].end + 0.1
                        }
                    phrases.append(phrase_obj)
                    current_phrase = []

        i += 1

    return phrases

def process_video_with_emoji_overlays(input_path: str, output_path: str, phrases: List[Dict], caption_style: str = 'default') -> None:
    """Process video with emoji overlays and updated caption positioning."""
    try:
        # Get video dimensions and duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            input_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        probe_data = json.loads(probe_result.stdout)
        stream_info = probe_data['streams'][0]
        video_width = int(stream_info['width'])
        video_height = int(stream_info['height'])
        video_duration = float(stream_info['duration'])

        # Adjust the end_time of the last phrase to match the video's duration
        if phrases:
            last_phrase = phrases[-1]
            last_phrase['end'] = max(last_phrase['end'], video_duration)
 
        # Create temporary ASS file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as temp_ass:
            temp_ass.write(create_ass_header(video_width, video_height, caption_style))
            temp_ass.write(create_ass_events(phrases, video_width, video_height, caption_style))
            temp_ass_path = temp_ass.name

        # Initialize emoji generator and mapper
        emoji_generator = EmojiImageGenerator()
        emoji_mapper = EmojiMapper()

        # Process emojis and create filter complex
        emoji_inputs = []
        filter_complex = []
        current = "0:v"
        emoji_count = 1


        for phrase in phrases:
            all_words = " ".join([
                word.word for word in (phrase.get('top_line', []) + phrase.get('bottom_line', []))
            ])

            keyword = emoji_mapper.extract_most_relevant_keyword(all_words, threshold=0.2)
            emoji_char = emoji_mapper.map_keyword_to_emoji(keyword)

            if emoji_char:
                emoji_path = emoji_generator.create_emoji_image(emoji_char)
                if emoji_path and os.path.exists(emoji_path):
                    emoji_inputs.extend(['-i', emoji_path])

                    # Determine emoji position based on caption layout
                    has_two_lines = bool(phrase.get('top_line') and phrase.get('bottom_line'))
                    y_position = 0.54 if has_two_lines else 0.54 # Higher for two lines, lower for single line

                    # Add emoji processing to filter complex
                    filter_complex.extend([
                        f"[{emoji_count}:v]scale=80:-1,format=rgba[emoji{emoji_count}]",
                        f"[{current}][emoji{emoji_count}]overlay="
                        f"x=(W-w)/2:y=(H*{y_position})"
                        f":enable='between(t,{float(phrase['start'])},{float(phrase['end'])})'"
                        f"[v{emoji_count}]"
                    ])
                    current = f"v{emoji_count}"
                    emoji_count += 1


        # Add ASS subtitles filter
        if filter_complex:  # Only add if we have emoji filters
            temp_ass_path_escaped = temp_ass_path.replace('\\', '/').replace(':', '\\:')
            filter_complex.append(f"[{current}]ass='{temp_ass_path_escaped}'[v]")
        else:
            # If no emojis were added, just use ASS filter directly
            temp_ass_path_escaped = temp_ass_path.replace('\\', '/').replace(':', '\\:')
            filter_complex = [f"[0:v]ass='{temp_ass_path_escaped}'[v]"]

        # Build FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            *emoji_inputs,
            '-filter_complex', ';'.join(filter_complex),
            '-map', '[v]',
            '-map', '0:a',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-c:a', 'copy',
            output_path
        ]

        # Execute FFmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        logger.info("Video processing completed successfully")

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise
    finally:
         
        if 'temp_ass_path' in locals():
            try:
                os.unlink(temp_ass_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary subtitle file: {e}")

# Implementing different caption styles



def create_ass_header(video_width: int, video_height: int, caption_style: str = 'default') -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Courier New,56,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1
"""

 
    if caption_style == 'fade_box':
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,80,&H000000,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,1,2,10,10,10,1
Style: Active,Arial,80,&H000000,&H000000FF,&H00FFFFFF,&H80000000,0,0,0,0,100,100,0,0,3,2,0,2,10,10,10,1
"""
    elif caption_style == 'karaoke':
        styles = """Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Highlight,Arial,80,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
"""
    elif caption_style == 'handwritten_outline':
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Brush Script MT,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,10,10,10,1
"""
    elif caption_style == 'gradient_text':
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,2,10,10,10,1
"""
    elif caption_style == 'typewriter':
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Courier New,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,10,10,10,1
"""
    elif caption_style == 'bounce_in':
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Verdana Bold,90,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
"""
    elif caption_style == 'glow_effect':
        styles = """Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
    """

    elif caption_style == 'submagic':
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
    """
    else:  # Default style
        styles = """Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
"""

    return header + styles + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
def apply_glow_per_word(words: List[WhisperWord], y_position: int,video_width) -> List[str]:
        word_events = []
        base_text = ' '.join(word.word for word in words)
        positions = []
        current_pos = 0
        for word in words:
            positions.append((current_pos, current_pos + len(word.word)))
            current_pos += len(word.word) + 1  # +1 for space

        for idx, word in enumerate(words):
            word_text = word.word
            start_idx, end_idx = positions[idx]
            text_with_effect = (
                base_text[:start_idx] +
                "{\\bord4\\3c&HADD8E6&}" + word_text + "{\\bord2\\3c&H000000&}" +
                base_text[end_idx:]
            )
            word_event = (
                f"Dialogue: 0,{format_ass_time(word.start)},{format_ass_time(word.end)},Default,,0,0,0,,"
                f"{{\\an8}}{{\\pos({video_width//2},{y_position})}}{text_with_effect}"
            )
            word_events.append(word_event)
        return word_events
def calculate_word_positions(words: List[WhisperWord], total_width: int, font_size: int = 56,
                           char_width_map: Dict[str, float] = None,
                           default_char_width: float = 0.44,
                           word_spacing: float = None) -> List[Dict]:
    """Calculate word positions with adjustable metrics for testing alignment."""
    if char_width_map is None:
        char_width_map = {
            'i': 0.23, 'l': 0.23, 'I': 0.23, '!': 0.23, '.': 0.23, ',': 0.23,
            ' ': 0.28,
            'm': 0.83, 'w': 0.83, 'M': 0.83, 'W': 0.83,
            'a': 0.44, 'b': 0.44, 'c': 0.44, 'd': 0.44, 'e': 0.44,
        }
    
    if word_spacing is None:
        word_spacing = font_size * 0.28

    def get_word_width(word: str) -> float:
        return sum(char_width_map.get(c, default_char_width) * font_size for c in word)
    
    word_positions = []
    total_text_width = sum(get_word_width(word.word) for word in words) + \
                      word_spacing * (len(words) - 1)
    
    start_x = (total_width - total_text_width) / 2
    current_x = start_x
    
    for i, word in enumerate(words):
        word_width = get_word_width(word.word)
        word_info = {
            'start': current_x,
            'center': current_x + (word_width / 2),
            'end': current_x + word_width,
            'width': word_width,
            'text': word.word,
            'word_obj': word,
            'text_width': word_width,
            'box_width': word_width + (font_size * 0.2),
            'box_height': font_size * 1.4
        }
        word_positions.append(word_info)
        current_x += word_width + (word_spacing if i < len(words) - 1 else 0)
    
    return word_positions
def type_text_sequence(word_list: List[WhisperWord], start_time: float, end_time: float, pos_x: int, pos_y: int, 
                      font_name: str, font_size: int) -> Tuple[List[str], float]:
    """Helper function to create typewriter sequence for words with dynamic typing speed."""
    events = []
    accumulated_text = ""
    current_time = start_time
    
    for i, word in enumerate(word_list):
        word_length = len(word.word)
        word_duration = word.end - word.start
        char_duration = word_duration / word_length if word_length > 0 else 0.05
        
        # Cancel previous event if it exists
        if events:
            events[-1] = events[-1].replace(
                format_ass_time(end_time),
                format_ass_time(current_time)
            )
        
        # Type each character in the word
        for j in range(word_length):
            char_time = current_time + (j * char_duration)
            accumulated_text += word.word[j]
            
            events.append(
                f"Dialogue: 0,{format_ass_time(char_time)},"
                f"{format_ass_time(end_time)},Default,,0,0,0,,"
                f"{{\\an7}}{{\\pos({pos_x},{pos_y})}}"
                f"{{\\fn{font_name}}}{{\\fs{font_size}}}"
                f"{accumulated_text}"
            )
        
        # Add space after word (except last word)
        if i < len(word_list) - 1:
            accumulated_text += " "
            current_time = char_time + char_duration
    
    return events, current_time
def apply_style_to_random_word(words: List[WhisperWord], base_text: str, style_override: str) -> str:
        word_list = base_text.split()
        random_word_idx = random.randint(0, len(word_list) - 1)
        # Apply style to a random word, then reset style
        styled_word = "{" + style_override + "}" + word_list[random_word_idx] + "{\\r}"
        word_list[random_word_idx] = styled_word
        return ' '.join(word_list)


def calculate_text_width(text: str, font_size: int) -> float:
    """Calculate approximate width of text for centering."""
    char_width_map = {
        'i': 0.3, 'l': 0.3, 'I': 0.3, '!': 0.3, '.': 0.3, ',': 0.3, "'": 0.3,
        'w': 0.9, 'm': 0.9, 'W': 0.9, 'M': 0.9,
        ' ': 0.3,
    }
    default_width = 0.5  # Default character width as proportion of font size
    
    total_width = sum(char_width_map.get(c, default_width) * font_size for c in text)
    return total_width

def create_ass_events(phrases: List[PhraseDict], video_width: int, video_height: int, caption_style: str = 'default') -> str:
    events: List[str] = []
    font_size = 56
    font_name = "Courier New"
    last_end_time = 0
    min_gap_between_phrases = 0.2


    TOP_Y = video_height * 0.62    # Single line or top line at 61%
    BOTTOM_Y = video_height * 0.67 # Bottom line at 67%

    if caption_style == 'typewriter':
        for phrase in phrases:
            start_time = max(phrase['start'], last_end_time + min_gap_between_phrases)
            end_time = phrase['end']
            
            if not (phrase['top_line'] or phrase['bottom_line']):
                continue
                
            last_end_time = end_time
            current_time = start_time

            # Calculate positions and text for each line independently
            if phrase['top_line']:
                full_top_text = " ".join(word.word for word in phrase['top_line'])
                top_text_width = calculate_text_width(full_top_text, font_size)
                top_start_x = video_width//2 - int(top_text_width//2)
                total_top_duration = sum((word.end - word.start) for word in phrase['top_line'])
                top_char_duration = total_top_duration / len(full_top_text)

                # Type out top line
                typed_top_text = ""
                for i in range(len(full_top_text)):
                    typed_top_text = full_top_text[:i+1]
                    char_start = current_time + (i * top_char_duration)
                    char_end = current_time + ((i + 1) * top_char_duration)
                    
                    events.append(
                        f"Dialogue: 0,{format_ass_time(char_start)},"
                        f"{format_ass_time(char_end)},Default,,0,0,0,,"
                        f"{{\\an7}}{{\\pos({top_start_x},{TOP_Y})}}"
                        f"{typed_top_text}"
                    )
                
                current_time += total_top_duration

            # Calculate and type bottom line independently
            if phrase['bottom_line']:
                y_position = BOTTOM_Y if phrase['top_line'] else TOP_Y
                full_bottom_text = " ".join(word.word for word in phrase['bottom_line'])
                bottom_text_width = calculate_text_width(full_bottom_text, font_size)
                bottom_start_x = video_width//2 - int(bottom_text_width//2)
                total_bottom_duration = sum((word.end - word.start) for word in phrase['bottom_line'])
                bottom_char_duration = total_bottom_duration / len(full_bottom_text)
                
                # Type out bottom line with separate events
                typed_bottom_text = ""
                for i in range(len(full_bottom_text)):
                    typed_bottom_text = full_bottom_text[:i+1]
                    char_start = current_time + (i * bottom_char_duration)
                    char_end = current_time + ((i + 1) * bottom_char_duration)
                    
                    # Keep top line visible if it exists
                    if phrase['top_line']:
                        events.append(
                            f"Dialogue: 0,{format_ass_time(char_start)},"
                            f"{format_ass_time(char_end)},Default,,0,0,0,,"
                            f"{{\\an7}}{{\\pos({top_start_x},{TOP_Y})}}"
                            f"{full_top_text}"
                        )
                    
                    # Add bottom line as separate event
                    events.append(
                        f"Dialogue: 0,{format_ass_time(char_start)},"
                        f"{format_ass_time(char_end)},Default,,0,0,0,,"
                        f"{{\\an7}}{{\\pos({bottom_start_x},{y_position})}}"
                        f"{typed_bottom_text}"
                    )
                
                current_time += total_bottom_duration

            # Final state - show complete text until end time
            if phrase['top_line']:
                events.append(
                    f"Dialogue: 0,{format_ass_time(current_time)},"
                    f"{format_ass_time(end_time)},Default,,0,0,0,,"
                    f"{{\\an7}}{{\\pos({top_start_x},{TOP_Y})}}"
                    f"{full_top_text}"
                )
            
            if phrase['bottom_line']:
                events.append(
                    f"Dialogue: 0,{format_ass_time(current_time)},"
                    f"{format_ass_time(end_time)},Default,,0,0,0,,"
                    f"{{\\an7}}{{\\pos({bottom_start_x},{y_position})}}"
                    f"{full_bottom_text}"
                )
    # Modified submagic style handler
    elif caption_style == 'submagic':
        for phrase in phrases:
            start_time = max(phrase['start'], last_end_time + min_gap_between_phrases)
            end_time = phrase['end']
            
            # Get all words and determine if we need to split into two lines
            all_words = []
            if phrase['top_line']:
                all_words.extend(phrase['top_line'])
            if phrase['bottom_line']:
                all_words.extend(phrase['bottom_line'])
                
            if not all_words:
                continue
                
            last_end_time = end_time

            # Determine if we have two lines
            has_two_lines = bool(phrase['top_line'] and phrase['bottom_line'])

            # Process each word state
            for current_word in all_words:
                word_duration = current_word.end - current_word.start
                half_duration = word_duration / 2
                
                # Build top line text with animation
                top_text = ""
                if phrase['top_line']:
                    for word in phrase['top_line']:
                        if word == current_word:
                            top_text += (
                                f"{{\\t({format_ass_time(current_word.start)},{format_ass_time(current_word.start + half_duration)},\\fscx120\\fscy120)"
                                f"\\t({format_ass_time(current_word.start + half_duration)},{format_ass_time(current_word.end)},\\fscx100\\fscy100)}}"
                                f"{{\\c&H00FF00&}}{word.word}{{\\c&HFFFFFF&}}"
                            )
                        else:
                            top_text += word.word
                        top_text += " "
                    top_text = top_text.rstrip()

                # Build bottom line text with animation
                bottom_text = ""
                if phrase['bottom_line']:
                    for word in phrase['bottom_line']:
                        if word == current_word:
                            bottom_text += (
                                f"{{\\t({format_ass_time(current_word.start)},{format_ass_time(current_word.start + half_duration)},\\fscx120\\fscy120)"
                                f"\\t({format_ass_time(current_word.start + half_duration)},{format_ass_time(current_word.end)},\\fscx100\\fscy100)}}"
                                f"{{\\c&H00FF00&}}{word.word}{{\\c&HFFFFFF&}}"
                            )
                        else:
                            bottom_text += word.word
                        bottom_text += " "
                    bottom_text = bottom_text.rstrip()

                # Create event with both lines
                event = (
                    f"Dialogue: 0,{format_ass_time(current_word.start)},"
                    f"{format_ass_time(current_word.end)},Default,,0,0,0,,"
                    f"{{\\an8}}"
                )
                
                if phrase['top_line']:
                    event += f"{{\\pos({video_width//2},{TOP_Y})}}{top_text}"
                if phrase['bottom_line']:
                    y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                    if phrase['top_line']:
                        event += "\\N"
                    event += f"{{\\pos({video_width//2},{y_pos})}}{bottom_text}"
                
                events.append(event)

            # Add base event for start period (before first word) - all white
            base_text = (
                f"Dialogue: 0,{format_ass_time(start_time)},"
                f"{format_ass_time(all_words[0].start)},Default,,0,0,0,,"
                f"{{\\an8}}"
            )
            if phrase['top_line']:
                base_text += f"{{\\pos({video_width//2},{TOP_Y})}}{' '.join(word.word for word in phrase['top_line'])}\\N"
            if phrase['bottom_line']:
                y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                base_text += f"{{\\pos({video_width//2},{y_pos})}}{' '.join(word.word for word in phrase['bottom_line'])}"
            events.append(base_text)

            # Add end event (after last word) - all white
            end_text = (
                f"Dialogue: 0,{format_ass_time(all_words[-1].end)},"
                f"{format_ass_time(end_time)},Default,,0,0,0,,"
                f"{{\\an8}}"
            )
            if phrase['top_line']:
                end_text += f"{{\\pos({video_width//2},{TOP_Y})}}{' '.join(word.word for word in phrase['top_line'])}\\N"
            if phrase['bottom_line']:
                y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                end_text += f"{{\\pos({video_width//2},{y_pos})}}{' '.join(word.word for word in phrase['bottom_line'])}"
            events.append(end_text)
    elif caption_style == 'glow_effect':
        for phrase in phrases:
            start_time = max(phrase['start'], last_end_time + min_gap_between_phrases)
            end_time = phrase['end']
            
            # Get all words and determine if we need to split into two lines
            all_words = []
            if phrase['top_line']:
                all_words.extend(phrase['top_line'])
            if phrase['bottom_line']:
                all_words.extend(phrase['bottom_line'])
                
            if not all_words:
                continue
                
            last_end_time = end_time

            # Split words into top and bottom lines if more than 3 words
            has_two_lines = bool(phrase['top_line'] and phrase['bottom_line'])
            if not has_two_lines:
                y_position = TOP_Y  # Single line at TOP_Y
            

            # Create base event for start period (before first word) - all white
            base_text = (
                f"Dialogue: 0,{format_ass_time(start_time)},"
                f"{format_ass_time(all_words[0].start)},Default,,0,0,0,,"
                f"{{\\an8}}"
            )
            if phrase['top_line']:
                base_text += f"{{\\pos({video_width//2},{TOP_Y})}}{' '.join(word.word for word in phrase['top_line'])}\\N"
            if phrase['bottom_line']:
                y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                base_text += f"{{\\pos({video_width//2},{y_pos})}}{' '.join(word.word for word in phrase['bottom_line'])}"
            events.append(base_text)

            # Process each word state
            for current_word in all_words:
                # Build top line text with animation
                top_text = ""
                if phrase['top_line']:
                    for word in phrase['top_line']:
                        if word == current_word:
                            top_text += (
                                f"{{\\bord4\\3c&HADD8E6&}}{word.word}{{\\bord2\\3c&H000000&}}"
                            )
                        else:
                            top_text += word.word
                        top_text += " "
                    top_text = top_text.rstrip()

                # Build bottom line text with animation
                bottom_text = ""
                if phrase['bottom_line']:
                    for word in phrase['bottom_line']:
                        if word == current_word:
                            bottom_text += (
                                f"{{\\bord4\\3c&HADD8E6&}}{word.word}{{\\bord2\\3c&H000000&}}"
                            )
                        else:
                            bottom_text += word.word
                        bottom_text += " "
                    bottom_text = bottom_text.rstrip()

                # Create event with both lines
                event = (
                    f"Dialogue: 0,{format_ass_time(current_word.start)},"
                    f"{format_ass_time(current_word.end)},Default,,0,0,0,,"
                    f"{{\\an8}}"
                )
                
                if phrase['top_line']:
                    event += f"{{\\pos({video_width//2},{TOP_Y})}}{top_text}"
                if phrase['bottom_line']:
                    y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                    if phrase['top_line']:
                        event += "\\N"
                    event += f"{{\\pos({video_width//2},{y_pos})}}{bottom_text}"
                
                events.append(event)

            # Add end event (after last word) - all white
            end_text = (
                f"Dialogue: 0,{format_ass_time(all_words[-1].end)},"
                f"{format_ass_time(end_time)},Default,,0,0,0,,"
                f"{{\\an8}}"
            )
            if phrase['top_line']:
                end_text += f"{{\\pos({video_width//2},{TOP_Y})}}{' '.join(word.word for word in phrase['top_line'])}\\N"
            if phrase['bottom_line']:
                y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                end_text += f"{{\\pos({video_width//2},{y_pos})}}{' '.join(word.word for word in phrase['bottom_line'])}"
            events.append(end_text)
    # Modified enlarge style handler
    elif caption_style == 'enlarge':
        for phrase in phrases:
            start_time = max(phrase['start'], last_end_time + min_gap_between_phrases)
            end_time = phrase['end']
            
            # Get all words and determine if we need to split into two lines
            all_words = []
            if phrase['top_line']:
                all_words.extend(phrase['top_line'])
            if phrase['bottom_line']:
                all_words.extend(phrase['bottom_line'])
                
            if not all_words:
                continue
                
            last_end_time = end_time

            # Determine if we have two lines
            has_two_lines = bool(phrase['top_line'] and phrase['bottom_line'])

            # Process each word state
            for current_word in all_words:
                # Build top line text with animation
                top_text = ""
                if phrase['top_line']:
                    for word in phrase['top_line']:
                        if word == current_word:
                            top_text += (
                                f"{{\\c&H00FF00&}}{{\\fscx120\\fscy120}}{word.word}{{\\r}}"
                            )
                        else:
                            top_text += word.word
                        top_text += " "
                    top_text = top_text.rstrip()

                # Build bottom line text with animation
                bottom_text = ""
                if phrase['bottom_line']:
                    for word in phrase['bottom_line']:
                        if word == current_word:
                            bottom_text += (
                                f"{{\\c&H00FF00&}}{{\\fscx120\\fscy120}}{word.word}{{\\r}}"
                            )
                        else:
                            bottom_text += word.word
                        bottom_text += " "
                    bottom_text = bottom_text.rstrip()

                # Create event with both lines
                event = (
                    f"Dialogue: 0,{format_ass_time(current_word.start)},"
                    f"{format_ass_time(current_word.end)},Default,,0,0,0,,"
                    f"{{\\an8}}"
                )
                
                if phrase['top_line']:
                    event += f"{{\\pos({video_width//2},{TOP_Y})}}{top_text}"
                if phrase['bottom_line']:
                    y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                    if phrase['top_line']:
                        event += "\\N"
                    event += f"{{\\pos({video_width//2},{y_pos})}}{bottom_text}"
                
                events.append(event)

            # Add base event for start period (before first word) - all white
            base_text = (
                f"Dialogue: 0,{format_ass_time(start_time)},"
                f"{format_ass_time(all_words[0].start)},Default,,0,0,0,,"
                f"{{\\an8}}"
            )
            if phrase['top_line']:
                base_text += f"{{\\pos({video_width//2},{TOP_Y})}}{' '.join(word.word for word in phrase['top_line'])}\\N"
            if phrase['bottom_line']:
                y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                base_text += f"{{\\pos({video_width//2},{y_pos})}}{' '.join(word.word for word in phrase['bottom_line'])}"
            events.append(base_text)

            # Add end event (after last word) - all white
            end_text = (
                f"Dialogue: 0,{format_ass_time(all_words[-1].end)},"
                f"{format_ass_time(end_time)},Default,,0,0,0,,"
                f"{{\\an8}}"
            )
            if phrase['top_line']:
                end_text += f"{{\\pos({video_width//2},{TOP_Y})}}{' '.join(word.word for word in phrase['top_line'])}\\N"
            if phrase['bottom_line']:
                y_pos = BOTTOM_Y if has_two_lines else TOP_Y
                end_text += f"{{\\pos({video_width//2},{y_pos})}}{' '.join(word.word for word in phrase['bottom_line'])}"
            events.append(end_text)
    elif caption_style == 'fade_box':
        for phrase in phrases:
            start_time = max(phrase['start'], last_end_time + min_gap_between_phrases)
            end_time = phrase['end']
            top_line = phrase['top_line']
            bottom_line = phrase['bottom_line']
            
            last_end_time = end_time
            
            # Determine if we have two lines
            has_two_lines = bool(top_line and bottom_line)
            y_position = TOP_Y if has_two_lines else TOP_Y

            # Process both lines together for two-line captions
            if has_two_lines:
                top_text = " ".join(word.word for word in top_line)
                bottom_text = " ".join(word.word for word in bottom_line)
                box_padding = "   "
                
                # Create combined background box for both lines
                combined_box_event = (
                    f"Dialogue: -1,{format_ass_time(start_time)},"
                    f"{format_ass_time(end_time)},Default,,0,0,0,,"
                    f"{{\\an8}}{{\\pos({video_width//2},{TOP_Y})}}"
                    f"{{\\3c&HDEDEDE&}}{{\\4c&HDEDEDE&}}"
                    f"{{\\bord8}}{{\\shad0}}{{\\4a&H20&}}"
                    f"{box_padding}{top_text}{box_padding}\\N"
                    f"{box_padding}{bottom_text}{box_padding}"
                )
                events.append(combined_box_event)
                
                # Process all words in sequence across both lines
                all_words = []
                for word in top_line:
                    all_words.append(('top', word))
                for word in bottom_line:
                    all_words.append(('bottom', word))
                
                # Sort words by start time to handle them in chronological order
                all_words.sort(key=lambda x: x[1].start)
                
                # Process each word while maintaining line position
                for i, (line_type, word) in enumerate(all_words):
                    # Separate words by line
                    current_top_text = ""
                    current_bottom_text = ""
                    
                    # Build current state of text
                    for j, (prev_line, prev_word) in enumerate(all_words):
                        if j <= i:
                            # Words up to current (black)
                            if prev_line == 'top':
                                current_top_text += f"{{\\c&H000000&}}{prev_word.word} "
                            else:
                                current_bottom_text += f"{{\\c&H000000&}}{prev_word.word} "
                        else:
                            # Words after current (gray)
                            if prev_line == 'top':
                                current_top_text += f"{{\\c&H808080&}}{prev_word.word} "
                            else:
                                current_bottom_text += f"{{\\c&H808080&}}{prev_word.word} "
                    
                    # Trim trailing spaces
                    current_top_text = current_top_text.rstrip()
                    current_bottom_text = current_bottom_text.rstrip()
                    
                    # Create event for this word state
                    word_event = (
                        f"Dialogue: 1,{format_ass_time(word.start)},"
                        f"{format_ass_time(word.end)},Default,,0,0,0,,"
                        f"{{\\an8}}{{\\pos({video_width//2},{TOP_Y})}}"
                        f"{{\\bord0}}{{\\shad0}}"
                        f"{current_top_text}\\N{current_bottom_text}"
                    )
                    events.append(word_event)
                    
            else:
                # Single line handling
                line_words = bottom_line if bottom_line else top_line
                if not line_words:
                    continue
                    
                full_text = " ".join(word.word for word in line_words)
                box_padding = "   "
                box_text = box_padding + full_text + box_padding
                
                # Background box event
                box_event = (
                    f"Dialogue: -1,{format_ass_time(line_words[0].start)},"
                    f"{format_ass_time(line_words[-1].end)},Default,,0,0,0,,"
                    f"{{\\an8}}{{\\pos({video_width//2},{y_position}}}"
                    f"{{\\3c&HDEDEDE&}}{{\\4c&HDEDEDE&}}"
                    f"{{\\bord8}}{{\\shad0}}{{\\4a&H20&}}"
                    f"{box_text}"
                )
                events.append(box_event)
                
                # Word-by-word color transition
                for i, word in enumerate(line_words):
                    colored_text = ""
                    if i > 0:
                        colored_text += f"{{\\c&H000000&}}" + " ".join(w.word for w in line_words[:i]) + " "
                    
                    colored_text += f"{{\\c&H000000&}}{word.word}"
                    
                    if i < len(line_words) - 1:
                        colored_text += f" {{\\c&H808080&}}" + " ".join(w.word for w in line_words[i+1:])
                    
                    word_event = (
                        f"Dialogue: 1,{format_ass_time(word.start)},"
                        f"{format_ass_time(word.end)},Default,,0,0,0,,"
                        f"{{\\an8}}{{\\pos({video_width//2},{y_position}}}"
                        f"{{\\bord0}}{{\\shad0}}"
                        f"{colored_text}"
                    )
                    events.append(word_event)
    else:
        for phrase_idx, phrase in enumerate(phrases):
            start_time = max(phrase['start'], last_end_time + min_gap_between_phrases)
            end_time = phrase['end']
            top_line = phrase['top_line']
            bottom_line = phrase['bottom_line']
            
            last_end_time = end_time

            # Determine text position
            has_two_lines = bool(top_line and bottom_line)
            if has_two_lines:
                top_y = TOP_Y
                bottom_y = BOTTOM_Y
            else:
                top_y = bottom_y = TOP_Y

            # Process each line (top and bottom) for all styles
            for line_type, line_words, y_position in [
                (0, top_line, top_y),
                (1, bottom_line, bottom_y)
            ]:
                if not line_words:
                    continue

                if caption_style == 'karaoke':
                    full_text = " ".join(word.word for word in line_words)
                    
                    # Base white text
                    base_event = (
                        f"Dialogue: 0,{format_ass_time(start_time)},"
                        f"{format_ass_time(end_time)},Default,,0,0,0,,"
                        f"{{\\an5}}{{\\pos({video_width//2},{y_position})}}"
                        f"{full_text}"
                    )
                    events.append(base_event)

                    # Highlight each word as it's spoken
                    for i, word in enumerate(line_words):
                        before_text = " ".join(w.word for w in line_words[:i])
                        after_text = " ".join(w.word for w in line_words[i+1:])
                        
                        if before_text: before_text += " "
                        if after_text: after_text = " " + after_text
                        
                        highlight_event = (
                            f"Dialogue: 1,{format_ass_time(word.start)},"
                            f"{format_ass_time(word.end)},Default,,0,0,0,,"
                            f"{{\\an5}}{{\\pos({video_width//2},{y_position})}}"
                            f"{before_text}{{\\c&H00FFFF&}}{word.word}{{\\c&HFFFFFF&}}{after_text}"
                        )
                        events.append(highlight_event)

   

                elif caption_style == 'gradient_text':
                    gradient_colors = ['&H00FF0000', '&H00FF7F00', '&H00FFFF00', '&H007FFF00',
                                '&H0000FF00', '&H0000FF7F', '&H0000FFFF', '&H00007FFF',
                                '&H000000FF', '&H007F00FF', '&H00FF00FF', '&H00FF007F']
                    color = random.choice(gradient_colors)
                    style_override = '\\c' + color
                    base_text = " ".join(word.word for word in line_words)
                    styled_text = apply_style_to_random_word(line_words, base_text, style_override)
                    events.append(
                        f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,"
                        f"{{\\an8}}{{\\pos({video_width//2},{y_position})}}{styled_text}"
                    )

                elif caption_style == 'bounce_in':
                    base_text = " ".join(word.word for word in line_words)
                    move_start_y = video_height + 100
                    animation_duration = min((end_time - start_time) * 1000 / 2, 500)
                    
                    bounce_y = y_position - 20
                    events.append(
                        f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(start_time + 0.3)},Default,,0,0,0,,"
                        f"{{\\an5}}{{\\move({video_width//2},{move_start_y},{video_width//2},{bounce_y},0,{int(animation_duration)})}}{{\\fs{font_size}}}{{\\t({int(animation_duration)},{int(animation_duration + 100)},\\fscx110\\fscy110)}}{base_text}"
                    )
                    events.append(
                        f"Dialogue: 0,{format_ass_time(start_time + 0.3)},{format_ass_time(end_time)},Default,,0,0,0,,"
                        f"{{\\an5}}{{\\move({video_width//2},{bounce_y},{video_width//2},{y_position},{int(animation_duration)},{int(animation_duration + 100)})}}{{\\fs{font_size}}}{{\\t(0,100,\\fscx100\\fscy100)}}{base_text}"
                    )

    return "\n".join(events)