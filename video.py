import os
from PIL import Image, ImageFilter
import numpy as np
from moviepy.editor import (
    ColorClip, ImageClip, CompositeVideoClip, VideoFileClip, 
    concatenate_videoclips, AudioFileClip, VideoClip
)
import shutil
import subprocess
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
def slide_in(clip, duration, side):
    w, h = clip.size
    pos_dict = {
        "left": lambda t: (min(0, w * (t / duration - 1)), "center"),
        "right": lambda t: (max(0, w * (1 - t / duration)), "center"),
        "top": lambda t: ("center", min(0, h * (t / duration - 1))),
        "bottom": lambda t: ("center", max(0, h * (1 - t / duration))),
    }
    return clip.set_position(pos_dict[side])
def optimize_worker_count(cpu_count):
    """Optimize the number of workers based on CPU count."""
    return max(1, min(cpu_count - 1, 4))
def slide_out(clip, duration, side):
    w, h = clip.size
    ts = clip.duration - duration
    pos_dict = {
        "left": lambda t: (min(0, w * (-(t - ts) / duration)), "center"),
        "right": lambda t: (max(0, w * ((t - ts) / duration)), "center"),
        "top": lambda t: ("center", min(0, h * (-(t - ts) / duration))),
        "bottom": lambda t: ("center", max(0, h * ((t - ts) / duration))),
    }
    return clip.set_position(pos_dict[side])

def simple_fade_transition(clip1, clip2, duration):
    """Simple fade transition as a fallback when complex transitions fail."""
    try:
        return CompositeVideoClip([
            clip1.set_duration(duration),
            clip2.set_duration(duration).crossfadein(duration)
        ]).set_duration(duration)
    except Exception as e:
        print(f"Fallback transition failed: {e}")
        return clip2.set_duration(duration)

def apply_video_effect(image_path: str, output_path: str, duration: float = 2.0):
    """Create video from image using direct FFmpeg command with specified duration."""
    try:
        # Direct FFmpeg command with dynamic duration
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-c:v', 'libx264',
            '-t', str(duration),  # Use the passed duration
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Verify the output duration
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            output_path
        ]
        actual_duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Verify the duration is within 0.1s tolerance
        if abs(actual_duration - duration) > 0.1:
            raise Exception(f"Generated video duration {actual_duration}s doesn't match requested {duration}s")
            
        return output_path
    except Exception as e:
        print(f"Error in apply_video_effect: {str(e)}")
        return None



def add_transition(clip1, clip2, transition_type, duration=2.0):
    """Refined transitions with smoother effects and additional options."""
    try:
        w, h = clip1.w, clip1.h

        # Ensure clips have valid durations
        if clip1.duration <= 0 or clip2.duration <= 0:
            warnings.warn("Invalid clip duration, falling back to simple fade transition.")
            return simple_fade_transition(clip1, clip2, duration)

        # Pre-cache frames for complex transitions with safer frame access
        try:
            if transition_type in ["zoom blend", "rotate fade", "zoom wipe", "reveal zoom", "spin", "burn", "horizontal banding", "diagonal soft wipe", "blinds"]:
                frame1_time = max(0, clip1.duration - duration)
                frame2_time = min(duration, clip2.duration)
                frame1 = clip1.get_frame(frame1_time)
                frame2 = clip2.get_frame(frame2_time)
        except Exception as e:
            warnings.warn(f"Frame pre-caching failed for '{transition_type}', falling back to fade transition: {e}")
            return simple_fade_transition(clip1, clip2, duration)

        if transition_type == "fade":
            return concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)], padding=-duration)

        elif transition_type == "wipe":
            def wipe(t):
                progress = t / duration
                x = -w * progress
                return (x, 0)

            clip1_out = clip1.set_position(wipe).set_end(duration)
            clip2_in = clip2.set_start(0).set_position((0, 0))
            return CompositeVideoClip([clip1_out, clip2_in], size=(w, h)).set_duration(duration)

        elif transition_type == "shaky wipe":
            def shaky_wipe(t):
                progress = t / duration
                x = -w * progress
                shake_amplitude = 10
                shake = shake_amplitude * np.sin(progress * np.pi * 10)
                return (x, shake)

            clip1_out = clip1.set_position(shaky_wipe).set_end(duration)
            clip2_in = clip2.set_start(0).set_position((0, 0))
            return CompositeVideoClip([clip1_out, clip2_in], size=(w, h)).set_duration(duration)

        elif transition_type == "zoom blend":
            def make_frame(t):
                progress = t / duration
                zoom_factor = 1 + 0.5 * progress

                frame1_zoomed = clip1.get_frame(clip1.duration - duration + t)
                frame1_zoomed = Image.fromarray(frame1_zoomed).resize((int(w * zoom_factor), int(h * zoom_factor)), Image.LANCZOS)
                frame1_zoomed = np.array(frame1_zoomed.crop((
                    (frame1_zoomed.width - w) // 2,
                    (frame1_zoomed.height - h) // 2,
                    (frame1_zoomed.width + w) // 2,
                    (frame1_zoomed.height + h) // 2
                )))

                frame2 = clip2.get_frame(t)

                blended_frame = frame1_zoomed * (1 - progress) + frame2 * progress
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "rotate fade":
            def make_frame(t):
                progress = t / duration
                angle = 360 * progress

                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                frame1_rotated = Image.fromarray(frame1).rotate(angle, resample=Image.BICUBIC, expand=False)
                frame1_rotated = frame1_rotated.resize((w, h), Image.LANCZOS)
                frame1_rotated = np.array(frame1_rotated)

                blended_frame = frame1_rotated * (1 - progress) + frame2 * progress
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "burn":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                burn_mask = np.clip(progress * 5, 0, 1)
                burn_color = np.array([255, 140, 0], dtype=np.uint8)

                blended_frame = frame1 * (1 - burn_mask) + burn_color * burn_mask + frame2 * burn_mask
                return np.clip(blended_frame, 0, 255).astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "horizontal banding":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                band_height = int(h * 0.05)
                num_bands = h // band_height
                frame = np.zeros_like(frame1)
                for i in range(num_bands):
                    y_start = i * band_height
                    y_end = y_start + band_height
                    if i % 2 == int(progress * num_bands) % 2:
                        frame[y_start:y_end, :, :] = frame2[y_start:y_end, :, :]
                    else:
                        frame[y_start:y_end, :, :] = frame1[y_start:y_end, :, :]
                return frame

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "diagonal soft wipe":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                mask = np.zeros((h, w), dtype=np.float32)
                X, Y = np.meshgrid(np.arange(w), np.arange(h))
                mask_area = (X + Y) / (w + h) < progress
                mask[mask_area] = 1
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
                mask = np.array(mask).astype(np.float32) / 255

                blended_frame = frame1 * (1 - mask[:, :, np.newaxis]) + frame2 * mask[:, :, np.newaxis]
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "blinds":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                num_slits = 10
                slit_height = h // num_slits
                frame = np.zeros_like(frame1)
                for i in range(num_slits):
                    y_start = i * slit_height
                    y_end = y_start + slit_height
                    if i % 2 == int(progress * num_slits) % 2:
                        frame[y_start:y_end, :, :] = frame2[y_start:y_end, :, :]
                    else:
                        frame[y_start:y_end, :, :] = frame1[y_start:y_end, :, :]
                return frame

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "spin":
            def make_frame(t):
                progress = t / duration
                angle = 360 * progress

                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                frame1_rotated = Image.fromarray(frame1).rotate(angle, resample=Image.BICUBIC, expand=False)
                frame1_rotated = frame1_rotated.resize((w, h), Image.LANCZOS)
                frame1_rotated = np.array(frame1_rotated)

                blended_frame = frame1_rotated * (1 - progress) + frame2 * progress
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "zoom wipe":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                zoom_factor = 1 + progress * 0.5
                frame1_zoomed = Image.fromarray(frame1).resize((int(w * zoom_factor), int(h * zoom_factor)), Image.LANCZOS)
                frame1_zoomed = np.array(frame1_zoomed.crop((
                    (frame1_zoomed.width - w) // 2,
                    (frame1_zoomed.height - h) // 2,
                    (frame1_zoomed.width + w) // 2,
                    (frame1_zoomed.height + h) // 2
                )))

                mask = np.zeros((h, w), dtype=np.float32)
                wipe_position = int(w * progress)
                mask[:, :wipe_position] = 1

                blended_frame = frame1_zoomed * (1 - mask[:, :, np.newaxis]) + frame2 * mask[:, :, np.newaxis]
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "reveal zoom":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                if progress < 0.5:
                    wipe_progress = progress * 2
                    wipe_position = int(w * wipe_progress)
                    mask = np.zeros((h, w), dtype=np.float32)
                    mask[:, :wipe_position] = 1
                    blended_frame = frame1 * (1 - mask[:, :, np.newaxis]) + frame1 * mask[:, :, np.newaxis]
                else:
                    zoom_progress = (progress - 0.5) * 2
                    zoom_factor = 1 + zoom_progress * 0.5
                    frame1_zoomed = Image.fromarray(frame1).resize((int(w * zoom_factor), int(h * zoom_factor)), Image.LANCZOS)
                    frame1_zoomed = np.array(frame1_zoomed.crop((
                        (frame1_zoomed.width - w) // 2,
                        (frame1_zoomed.height - h) // 2,
                        (frame1_zoomed.width + w) // 2,
                        (frame1_zoomed.height + h) // 2
                    )))
                    blend_factor = (progress - 0.5) * 2
                    blended_frame = frame1_zoomed * (1 - blend_factor) + frame2 * blend_factor

                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        # New transitions
        elif transition_type == "circle reveal":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                mask = np.zeros((h, w), dtype=np.float32)
                center = (w // 2, h // 2)
                radius = int(np.sqrt(w**2 + h**2) * progress)
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                mask[dist_from_center <= radius] = 1
                mask = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=10))
                mask = np.array(mask).astype(np.float32) / 255

                blended_frame = frame1 * (1 - mask[:, :, np.newaxis]) + frame2 * mask[:, :, np.newaxis]
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "pixelate in":
            def make_frame(t):
                progress = t / duration
                frame2 = clip2.get_frame(t)

                scale = max(1, int(50 * (1 - progress)))
                small_frame = Image.fromarray(frame2).resize((w // scale, h // scale), Image.NEAREST)
                pixelated_frame = small_frame.resize((w, h), Image.NEAREST)

                return np.array(pixelated_frame)

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "color wipe":
            def make_frame(t):
                progress = t / duration
                frame2 = clip2.get_frame(t)

                color = np.array([255 * progress, 0, 255 * (1 - progress)], dtype=np.uint8)
                color_frame = np.full_like(frame2, color)

                blended_frame = frame2 * progress + color_frame * (1 - progress)
                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "page curl":
            # Simplified version due to complexity
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                curl_position = int(w * progress)
                frame = np.copy(frame1)
                frame[:, curl_position:, :] = frame2[:, curl_position:, :]

                return frame

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "flash":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                if progress < 0.5:
                    intensity = 1 - 2 * progress
                else:
                    intensity = (2 * progress) - 1

                flash_color = np.full_like(frame1, 255 * intensity)
                blended_frame = frame1 * (1 - intensity) + flash_color * intensity
                if progress > 0.5:
                    blended_frame = frame2 * (1 - intensity) + flash_color * intensity

                return np.clip(blended_frame, 0, 255).astype('uint8')

            return VideoClip(make_frame, duration=duration)

        else:
            warnings.warn(f"Transition '{transition_type}' not recognized. Using simple fade transition.")
            return simple_fade_transition(clip1, clip2, duration)

    except Exception as e:
        warnings.warn(f"Transition creation failed for '{transition_type}', falling back to simple fade transition: {e}")
        return simple_fade_transition(clip1, clip2, duration)


def get_audio_duration(audio_file):
    """Get the duration of an audio file."""
    try:
        audio = AudioFileClip(audio_file)
        duration = audio.duration
        audio.close()
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 0

def concatenate_videos_with_transitions(video_paths, audio_file, output_path, transitions=None, transition_duration=1.0):
    """Concatenate videos with transitions and ensure proper audio sync."""
    if not transitions:
        transitions = ['fade'] * (len(video_paths) - 1)
    
    # Ensure we have the right number of transitions
    while len(transitions) < len(video_paths) - 1:
        transitions.append('fade')
    
    try:
        # Load videos
        video_clips = []
        for path in video_paths:
            clip = VideoFileClip(path)
            # Ensure each clip has a minimum duration
            if clip.duration < 2.0:
                clip = clip.set_duration(2.0)
            video_clips.append(clip)

        # Apply transitions between clips
        final_clips = []
        for i in range(len(video_clips)):
            if i == len(video_clips) - 1:
                final_clips.append(video_clips[i])
                continue
                
            current_clip = video_clips[i]
            next_clip = video_clips[i + 1]
            transition_type = transitions[i]
            
            # Create transition
            try:
                transition = add_transition(
                    current_clip,
                    next_clip,
                    transition_type,
                    duration=transition_duration
                )
                
                # Set clip duration to match audio segment
                clip_duration = 2.0  # Default duration
                if transition:
                    clip_with_transition = CompositeVideoClip([
                        current_clip.set_duration(clip_duration),
                        transition.set_start(clip_duration - transition_duration)
                    ])
                    final_clips.append(clip_with_transition)
                else:
                    final_clips.append(current_clip.set_duration(clip_duration))
                    
            except Exception as e:
                print(f"Transition failed: {str(e)}, using fade transition")
                transition = add_transition(
                    current_clip,
                    next_clip,
                    'fade',
                    duration=transition_duration
                )
                if transition:
                    final_clips.append(current_clip.set_duration(clip_duration))

        # Concatenate all clips
        final_video = concatenate_videoclips(final_clips, method="compose")
        
        # Add audio
        audio = AudioFileClip(audio_file)
        final_video = final_video.set_audio(audio)
        
        # Write final video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=30,
            preset='ultrafast'
        )
        
        # Clean up clips
        for clip in video_clips:
            clip.close()
        for clip in final_clips:
            if hasattr(clip, 'close'):
                clip.close()
        audio.close()
        final_video.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error in video concatenation: {str(e)}")
        raise
def create_videos(image_files, effect_prompts, max_workers=None, output_path=None, duration=2.0):
    """Create videos with improved error handling and file ordering."""
    if not image_files:
        print("No image files provided")
        return []

    video_files = []
    file_order = {i+1: None for i in range(len(image_files))}
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, image in enumerate(image_files, 1):
                if not os.path.exists(image):
                    print(f"Image file not found: {image}")
                    continue
                    
                current_output = output_path if output_path else os.path.join("videos", f"video{i}.mp4")
                
                if os.path.exists(current_output):
                    try:
                        os.remove(current_output)
                    except Exception as e:
                        print(f"Warning: Could not remove existing video {current_output}: {e}")
                
                future = executor.submit(
                    apply_video_effect,
                    image,
                    current_output,
                    duration  # Pass through the duration parameter
                )
                futures[future] = (i, current_output)
            
            for future in as_completed(futures):
                idx, video_path = futures[future]
                try:
                    result = future.result()
                    if result and os.path.exists(result):
                        # Verify video integrity
                        cmd = [
                            'ffprobe',
                            '-v', 'error',
                            '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1',
                            result
                        ]
                        actual_duration = float(subprocess.check_output(cmd).decode().strip())
                        
                        if abs(actual_duration - duration) < 0.1:  # Allow 0.1s tolerance
                            file_order[idx] = result
                            print(f"Successfully created video {idx}: {result} (duration: {actual_duration}s)")
                        else:
                            print(f"Invalid duration for video {idx}: {actual_duration}s")
                            if os.path.exists(result):
                                os.remove(result)
                    else:
                        print(f"Failed to create video {idx}")
                except Exception as e:
                    print(f"Error creating video {idx}: {str(e)}")
        
        # Maintain original order
        video_files = [file_order[i] for i in sorted(file_order.keys()) if file_order[i] is not None]
        
        print(f"Successfully created {len(video_files)} videos")
        return video_files
        
    except Exception as e:
        print(f"Error in create_videos: {str(e)}")
        return []