from pathlib import Path

import cv2
import numpy as np
import tqdm


def reduce_video_frames(input_path, output_path, reduction_factor=2, keep_audio=False):
    """
    Reduce the number of frames in a video by keeping every nth frame.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save the output video
        reduction_factor (int): Keep every nth frame (e.g., 2 means keep every other frame)
        keep_audio (bool): Whether to keep the audio track (requires moviepy if True)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return False

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate new fps
        new_fps = original_fps / reduction_factor
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for .mp4
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            10,
            (frame_width, frame_height)
        )
        
        if not out.isOpened():
            print("Error: Could not create output video file")
            return False

        frame_count = 0
        frames_written = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep only every nth frame
            if frame_count % reduction_factor == 0:
                out.write(frame)
                frames_written += 1
            
            frame_count += 1
            
            progress_bar = tqdm.tqdm(total=total_frames, position=0, leave=True)
            progress_bar.set_description(f"Processing video: {frames_written} frames written")
            progress_bar.update(1)


        # Release resources
        cap.release()
        out.release()

        print(f"Video processing complete:")
        print(f"Original frames: {total_frames}")
        print(f"Original FPS: {original_fps}")
        print(f"New frames: {frames_written}")
        print(f"New FPS: {new_fps}")
        
        # Handle audio if requested
        if keep_audio:
            try:
                from moviepy.editor import AudioFileClip, VideoFileClip

                # Load the original video with audio
                original_video = VideoFileClip(input_path)
                
                if original_video.audio is not None:
                    # Load the new video
                    new_video = VideoFileClip(output_path)
                    
                    # Get the original audio
                    audio = original_video.audio
                    
                    # Create temporary path for the final video
                    temp_output = str(Path(output_path).with_name('temp_' + Path(output_path).name))
                    
                    # Combine new video with original audio
                    final_video = new_video.set_audio(audio)
                    final_video.write_videofile(
                        temp_output,
                        codec='libx264',
                        audio_codec='aac'
                    )
                    
                    # Clean up
                    new_video.close()
                    original_video.close()
                    
                    # Replace original output with the version with audio
                    import os
                    os.replace(temp_output, output_path)
                    
                    print("Audio track successfully preserved")
            except ImportError:
                print("Warning: moviepy not installed. Audio preservation skipped.")
                print("To keep audio, install moviepy: pip install moviepy")
            except Exception as e:
                print(f"Error preserving audio: {str(e)}")
        
        return True

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False

def batch_reduce_videos(input_dir, output_dir, reduction_factor=2, keep_audio=False):
    """
    Reduce frame count for all videos in a directory.
    
    Args:
        input_dir (str): Input directory containing videos
        output_dir (str): Output directory for processed videos
        reduction_factor (int): Keep every nth frame
        keep_audio (bool): Whether to keep the audio track
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    
    # Process each video file
    for video_file in input_path.iterdir():
        if video_file.suffix.lower() in video_extensions:
            output_file = output_path / video_file.name
            print(f"\nProcessing {video_file.name}...")
            
            success = reduce_video_frames(
                str(video_file),
                str(output_file),
                reduction_factor,
                keep_audio
            )
            
            if success:
                print(f"Successfully processed {video_file.name}")
            else:
                print(f"Failed to process {video_file.name}")

# Example usage
if __name__ == "__main__":
    # For single video
    input_video = "/fe8k-data/sensornet/Tue_2017-03-14_073002.avi"
    output_video = "Tue_2017-03-14_073002_reduced_fps.avi"
    reduce_video_frames(input_video, output_video, reduction_factor=10, keep_audio=True)
    
    # # For batch processing
    # input_directory = "path/to/input/videos"
    # output_directory = "path/to/output/videos"
    # batch_reduce_videos(input_directory, output_directory, reduction_factor=2, keep_audio=True)