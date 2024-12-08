from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import os

def add_background_music(video_path, music_path, output_path, music_volume=0.4):
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
        
    if not os.path.exists(music_path):
        print(f"Music file not found: {music_path}")
        return False
    
    try:
        # Load the video
        video = VideoFileClip(video_path)
        
        # Get the original audio from the video
        original_audio = video.audio
        if original_audio is None:
            print("Warning: Video has no audio track")
            return False
        
        # Load the background music
        background_music = AudioFileClip(music_path)
        
        # Loop the background music if it's shorter than the video
        if background_music.duration < video.duration:
            repeats = int(video.duration / background_music.duration) + 1
            background_music = background_music.loop(repeats)  # Changed from audio_loop to loop
            
        # Trim the background music if it's longer than the video
        background_music = background_music.subclip(0, video.duration)
        
        # Set the volume of the background music
        background_music = background_music.volumex(music_volume)
        
        # Combine the original audio with the background music
        final_audio = CompositeAudioClip([original_audio, background_music])
        
        # Set the final audio to the video
        final_video = video.set_audio(final_audio)
        
        # Write the final video file
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset='ultrafast'  # Added for faster processing
        )
        
        return True
        
    except Exception as e:
        print(f"Error adding background music: {str(e)}")
        return False
        
    finally:
        # Close all clips in finally block to ensure cleanup
        try:
            if 'video' in locals():
                video.close()
            if 'background_music' in locals():
                background_music.close()
            if 'final_video' in locals():
                final_video.close()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")