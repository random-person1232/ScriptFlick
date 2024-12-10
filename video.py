import os
from PIL import Image, ImageFilter
import numpy as np
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.editor import (
    ColorClip, ImageClip, CompositeVideoClip, VideoFileClip, 
    concatenate_videoclips, AudioFileClip, VideoClip
)
import shutil
import subprocess
import time
import math
import json
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

def init_clip(clip):
    """Initialize clip and its reader."""
    try:
        # Initialize the clip if needed
        if clip is None:
            return False
            
        # Initialize the reader
        if not hasattr(clip, 'reader') or clip.reader is None:
            # Try to read a frame to initialize
            _ = clip.get_frame(0)
            return True
            
        return True
    except Exception as e:
        print(f"Clip initialization failed: {str(e)}")
        return False
def add_transition(clip1, clip2, transition_type, duration=1.0):
    """Refined transitions with improved frame handling."""
    try:
        w, h = clip1.w, clip1.h
        
        if clip1.duration <= 0 or clip2.duration <= 0:
            print("Invalid clip duration, using fade transition")
            return simple_fade_transition(clip1, clip2, duration)


        # Initialize both clips
        if not init_clip(clip1) or not init_clip(clip2):
            print("Clip initialization failed, using simple fade")
            return simple_fade_transition(clip1, clip2, duration)

        # Replace transition creation like this:
        if transition_type in ["fade", "crossfade"]:
            clip1_part = clip1.set_duration(duration)
            clip2_part = clip2.set_duration(duration).crossfadein(duration)
            return concatenate_videoclips([clip1_part, clip2_part]).set_duration(duration)
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

        elif transition_type == "diagonal soft wipe":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                # Create diagonal gradient with smooth edges
                mask = np.zeros((h, w), dtype=np.float32)
                X, Y = np.meshgrid(np.arange(w), np.arange(h))
                
                # Improved diagonal calculation with adjustable angle
                angle = np.pi / 4  # 45 degrees
                diagonal_progress = (X * np.cos(angle) + Y * np.sin(angle)) / (w * np.cos(angle) + h * np.sin(angle))
                mask = (diagonal_progress < progress).astype(np.float32)
                
                # Apply gaussian blur for smoother edge
                blur_radius = int(min(w, h) * 0.05)  # Adaptive blur radius
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                mask = np.array(mask).astype(np.float32) / 255

                # Add subtle gradient overlay
                gradient = np.clip((diagonal_progress - progress + 0.1) * 3, 0, 1)
                gradient = Image.fromarray((gradient * 255).astype(np.uint8))
                gradient = gradient.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                gradient = np.array(gradient).astype(np.float32) / 255

                # Combine frames with enhanced masking
                mask = mask[:, :, np.newaxis]
                gradient = gradient[:, :, np.newaxis] * 0.2  # Subtle gradient effect
                blended_frame = frame1 * (1 - mask) * (1 - gradient) + frame2 * mask

                return blended_frame.astype('uint8')

            return VideoClip(make_frame, duration=duration)

        elif transition_type == "circle reveal":
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                # Create circular mask with improved edge handling
                mask = np.zeros((h, w), dtype=np.float32)
                Y, X = np.ogrid[:h, :w]
                
                # Calculate center and maximum radius
                center = (w // 2, h // 2)
                max_radius = np.sqrt(w**2 + h**2)
                
                # Create circle mask with smooth edges
                radius = max_radius * progress
                dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                
                # Add turbulence to edge for more organic feel
                turbulence = np.random.normal(0, max_radius * 0.01, dist_from_center.shape)
                dist_from_center += turbulence
                
                # Create soft edge
                edge_width = max_radius * 0.1
                mask = 1 - np.clip((dist_from_center - radius) / edge_width, 0, 1)
                
                # Apply gaussian blur for even smoother transition
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask = mask.filter(ImageFilter.GaussianBlur(radius=int(edge_width * 0.5)))
                mask = np.array(mask).astype(np.float32) / 255

                # Add subtle glow effect
                glow = mask.copy()
                glow = Image.fromarray((glow * 255).astype(np.uint8))
                glow = glow.filter(ImageFilter.GaussianBlur(radius=int(edge_width)))
                glow = np.array(glow).astype(np.float32) / 255
                
                # Combine frames with mask and glow
                mask = mask[:, :, np.newaxis]
                glow = glow[:, :, np.newaxis] * 0.2  # Subtle glow intensity
                blended_frame = frame1 * (1 - mask) + frame2 * mask + glow * 255

                return np.clip(blended_frame, 0, 255).astype('uint8')

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
            def make_frame(t):
                progress = t / duration
                frame1 = clip1.get_frame(clip1.duration - duration + t)
                frame2 = clip2.get_frame(t)

                # Create mesh grid for 3D effect
                X, Y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
                
                # Calculate curl parameters
                curl_amount = progress * np.pi
                curl_radius = w * 0.3  # Adjustable radius of curl
                shadow_width = int(w * 0.1)  # Width of shadow effect
                
                # Calculate page curl transformation
                x_offset = w * (1 - progress)
                curl_edge = x_offset + curl_radius * np.sin(curl_amount)
                
                # Create mask for page regions
                mask = np.zeros((h, w), dtype=np.float32)
                shadow_mask = np.zeros((h, w), dtype=np.float32)
                
                for i in range(w):
                    for j in range(h):
                        if i < x_offset:
                            # Flat part of page
                            mask[j, i] = 1
                        elif i < curl_edge:
                            # Curled part calculation
                            curl_progress = (i - x_offset) / curl_radius
                            curl_angle = curl_progress * curl_amount
                            
                            # Add perspective distortion
                            perspective = 1 - (curl_progress * 0.3)
                            y_distort = j * perspective + (h * (1 - perspective) / 2)
                            
                            if 0 <= y_distort < h:
                                mask[j, i] = np.cos(curl_angle) * perspective
                                
                                # Add shadow near curl
                                if i > x_offset + shadow_width:
                                    shadow_intensity = 1 - ((i - (x_offset + shadow_width)) / shadow_width)
                                    shadow_mask[j, i] = max(0, shadow_intensity * 0.5)

                # Expand masks to 3 channels for RGB
                mask = mask[:, :, np.newaxis]
                shadow_mask = shadow_mask[:, :, np.newaxis]
                
                # Combine frames with curl effect
                result = frame2 * (1 - mask) + frame1 * mask
                
                # Apply shadow effect
                shadow_color = np.array([0, 0, 0], dtype=np.uint8)
                result = result * (1 - shadow_mask) + shadow_color * shadow_mask
                
                return result.astype('uint8')

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


def simple_fade_transition(clip1, clip2, duration):
    """Reliable fallback transition."""
    try:
        # Initialize both clips' readers
        init_clip(clip1)
        init_clip(clip2)
        
        # Simple concatenation with crossfade
        return concatenate_videoclips([
            clip1.set_duration(duration),
            clip2.set_duration(duration).crossfadein(duration)
        ]).set_duration(duration)
    except Exception as e:
        print(f"Simple fade transition failed: {str(e)}")
        return clip2.set_duration(duration)
def apply_video_effect(image_path: str, output_path: str, duration: float = 2.0):
    """Create video from image with proper frame handling and buffer checks."""
    try:
        # Enhanced FFmpeg command with frame count and buffer size settings
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-t', str(duration),
            '-vf', f'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-tune', 'stillimage',  # Optimize for still image input
            '-pix_fmt', 'yuv420p',
            '-b:v', '2M',
            '-bufsize', '2M',      # Explicit buffer size
            '-maxrate', '2M',      # Maximum bitrate
            '-r', '30',            # Frame rate
            '-frames:v', str(int(duration * 30)),  # Exact frame count
            '-avoid_negative_ts', 'make_zero',
            '-movflags', '+faststart',
            output_path
        ]
        
        # Run FFmpeg with proper error handling
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
            
        # Verify the output
        if not os.path.exists(output_path):
            return None
            
        # Verify exact duration and frame count
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,duration',
            '-of', 'json',
            output_path
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            actual_frames = int(probe_data['streams'][0]['nb_frames'])
            expected_frames = int(duration * 30)
            
            if actual_frames != expected_frames:
                print(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
                return None
                
        return output_path
        
    except Exception as e:
        print(f"Error in apply_video_effect: {str(e)}")
        return None

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
    """Concatenate videos with proper transitions and timing."""
    if not transitions:
        transitions = ['fade'] * (len(video_paths) - 1)
    
    while len(transitions) < len(video_paths) - 1:
        transitions.append('fade')
    
    try:
        # Load videos and audio
        video_clips = []
        for path in video_paths:
            clip = VideoFileClip(path)
            if clip.duration < 2.0:
                clip = clip.set_duration(2.0)
            init_clip(clip)  # Initialize reader immediately
            video_clips.append(clip)

        # Get audio duration to ensure video matches
        audio = AudioFileClip(audio_file)
        total_audio_duration = audio.duration
        
        # Calculate required video duration
        num_clips = len(video_clips)
        num_transitions = num_clips - 1
        total_transition_time = num_transitions * transition_duration
        base_clip_duration = (total_audio_duration - total_transition_time) / num_clips

        # Process clips with transitions
        final_clips = []
        for i in range(len(video_clips)):
            # Set each clip to the calculated duration
            current_clip = video_clips[i].set_duration(base_clip_duration)
            
            if i == len(video_clips) - 1:
                final_clips.append(current_clip)
                continue
            
            next_clip = video_clips[i + 1].set_duration(base_clip_duration)
            
            # Create transition using the specified transition type
            transition = add_transition(
                current_clip,
                next_clip,
                transitions[i],
                transition_duration
            )
            
            if transition:
                # Add main clip (shortened to account for transition)
                final_clips.append(
                    current_clip.set_duration(base_clip_duration - transition_duration/2)
                )
                # Add transition
                final_clips.append(transition)
            else:
                # Fallback to full clip if transition fails
                final_clips.append(current_clip)

        # Concatenate all clips
        final_video = concatenate_videoclips(final_clips)
        
        # Add audio
        final_video = final_video.set_audio(audio)
        
        # Write final video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=30,
            preset='medium',
            bitrate='2M',
            write_logfile=True
        )
        
        # Clean up
        for clip in video_clips:
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