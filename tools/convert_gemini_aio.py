import time
import os
import glob
import json
import subprocess
import cv2
from pathlib import Path
import re
from PIL import Image
from google import genai
import random
import math
from tqdm import tqdm
import pandas as pd
# Initialize the Gemini client
client = genai.Client(api_key="")  # Replace with your actual key

def check_ffmpeg_installed():
    """Check if /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg is installed and available."""
    try:
        subprocess.run(['/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

# Global flag for /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg availability
FFMPEG_AVAILABLE = check_ffmpeg_installed()
if not FFMPEG_AVAILABLE:
    print("WARNING: /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg is not installed or not in PATH. Using fallback methods for video processing.")
    print("For best results, install /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg: https:///vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg.org/download.html")

def extract_first_frame(video_path, output_path):
    """Extract the first frame of a video and save it as JPG."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()
    return ret

def get_video_duration(video_path):
    """Get the duration of a video in seconds."""
    if FFMPEG_AVAILABLE:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        try:
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            return float(output)
        except (subprocess.CalledProcessError, ValueError):
            print(f"Error using ffprobe, falling back to OpenCV for {video_path}")

    # Fallback to OpenCV (less accurate but more compatible)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else None
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration with OpenCV: {str(e)}")
        return None

def has_audio_stream(video_path):
    """Check if the video has an audio stream."""
    if FFMPEG_AVAILABLE:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        try:
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            return bool(output)
        except subprocess.CalledProcessError:
            print(f"Error checking audio with ffprobe for {video_path}")

    # Without /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg, assume no audio to be safe
    return False

def copy_video(input_path, output_path):
    """Simple copy of video file when processing isn't available."""
    try:
        import shutil
        shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"Error copying video: {str(e)}")
        return False

def adjust_video_speed(video_path, output_path, target_duration=3.0):
    """Adjust video speed to make it match target_duration seconds."""
    # If /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg is not available, just copy the video
    if not FFMPEG_AVAILABLE:
        print(f"/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg not available, copying video without speed adjustment: {video_path}")
        return copy_video(video_path, output_path)

    original_duration = get_video_duration(video_path)
    if original_duration is None:
        print(f"Could not determine duration of {video_path}, copying without adjustment")
        return copy_video(video_path, output_path)

    # Calculate the speed factor
    if target_duration is None:
        speed_factor = 1.0
    else:
        speed_factor = original_duration / target_duration

    # Create a temporary file for preprocessing if needed
    temp_path = output_path + ".temp.mp4"

    # First ensure the video is properly encoded for API compatibility
    # This step also helps with connection issues
    preprocess_cmd = [
        '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-profile:v', 'main',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-y',
        temp_path
    ]

    try:
        subprocess.run(preprocess_cmd, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing video: {e.stderr.decode('utf-8')}")
        return copy_video(video_path, output_path)

    has_audio = has_audio_stream(temp_path)

    # For speeding up videos (original_duration > target_duration)
    if speed_factor > 1.0:
        if has_audio:
            # For extreme speedup, use multiple atempo filters (max 2.0 per filter)
            atempo_chain = ""
            temp_factor = speed_factor

            while temp_factor > 1.0:
                if temp_factor > 2.0:
                    atempo_chain += f"atempo=2.0,"
                    temp_factor /= 2.0
                else:
                    atempo_chain += f"atempo={temp_factor}"
                    temp_factor = 1.0

            # Remove trailing comma if needed
            if atempo_chain.endswith(','):
                atempo_chain = atempo_chain[:-1]

            cmd = [
                '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                '-i', temp_path,
                '-filter_complex', f'[0:v]setpts={1/speed_factor}*PTS[v];[0:a]{atempo_chain}[a]',
                '-map', '[v]',
                '-map', '[a]',
                '-y',
                output_path
            ]
        else:
            cmd = [
                '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                '-i', temp_path,
                '-filter:v', f'setpts={1/speed_factor}*PTS',
                '-an',
                '-y',
                output_path
            ]
    # For slowing down videos (original_duration < target_duration)
    else:
        # Handle extreme slowdowns (atempo values below 0.5)
        if speed_factor < 0.5:
            # Calculate how many times we need to apply atempo=0.5
            # For example, to get 0.25, we need to apply atempo=0.5 twice
            num_iterations = math.ceil(math.log(speed_factor) / math.log(0.5))
            individual_factor = speed_factor ** (1.0 / num_iterations)

            # Ensure we're above the minimum 0.5
            individual_factor = max(0.5, individual_factor)

            if has_audio:
                atempo_chain = f"atempo={individual_factor}"
                for _ in range(num_iterations - 1):
                    atempo_chain += f",atempo={individual_factor}"

                cmd = [
                    '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                    '-i', temp_path,
                    '-filter_complex', f'[0:v]setpts={1/speed_factor}*PTS[v];[0:a]{atempo_chain}[a]',
                    '-map', '[v]',
                    '-map', '[a]',
                    '-y',
                    output_path
                ]
            else:
                cmd = [
                    '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                    '-i', temp_path,
                    '-filter:v', f'setpts={1/speed_factor}*PTS',
                    '-an',
                    '-y',
                    output_path
                ]
        else:
            if has_audio:
                cmd = [
                    '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                    '-i', temp_path,
                    '-filter_complex', f'[0:v]setpts={1/speed_factor}*PTS[v];[0:a]atempo={speed_factor}[a]',
                    '-map', '[v]',
                    '-map', '[a]',
                    '-y',
                    output_path
                ]
            else:
                cmd = [
                    '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                    '-i', temp_path,
                    '-filter:v', f'setpts={1/speed_factor}*PTS',
                    '-an',
                    '-y',
                    output_path
                ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adjusting video speed: {e.stderr.decode('utf-8')}")
        # Try a simpler approach if complex filter fails
        try:
            simple_cmd = [
                '/vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg',
                '-i', temp_path,
                '-filter:v', f'setpts={1/speed_factor}*PTS',
                '-an',  # No audio
                '-y',
                output_path
            ]
            subprocess.run(simple_cmd, check=True, stderr=subprocess.PIPE)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Used simplified approach (no audio) for {video_path}")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"Error with simplified approach: {e2.stderr.decode('utf-8')}")
            # If all else fails, just copy the preprocessed file
            try:
                import shutil
                shutil.copy(temp_path, output_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"Copied preprocessed file for {video_path} (no speed adjustment)")
                return True
            except Exception as e3:
                print(f"Final fallback failed: {str(e3)}")
                return False

def process_image_video_pair(image_path, video_path, max_retries=3):
    """Process a single image-video pair and generate Venom-specific descriptions."""
    for attempt in range(max_retries):
        try:
            # Upload the video file to Gemini
            print(f"Uploading video: {video_path} (attempt {attempt+1}/{max_retries})")
            video_file = client.files.upload(file=video_path)

            # Wait for video processing to complete - reduced initial wait time
            wait_time = 3  # Reduced from 10s to 3s for faster processing
            while video_file.state.name == "PROCESSING":
                print(f"Waiting for video to be processed... ({wait_time}s)")
                time.sleep(wait_time)
                video_file = client.files.get(name=video_file.name)
                # Increase wait time gradually (but not too much)
                wait_time = min(wait_time * 1.5, 30)  # Also reduced max wait to 30s

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed for {video_path}")

            print(f"Video processing complete: {video_path}")

            # Load the image
            image = Image.open(image_path)

            # Updated prompt with our revised version
            prompt = """
            You're analyzing scenes from Venom movies showing symbiote transformations. The image is the first frame of the video. Please provide TWO separate, standalone descriptions:

            1. IMAGE DESCRIPTION:
            Describe this first frame in detail, showing the initial state of a Venom symbiote transformation. Capture the subject (human, animal, or object), the symbiote's appearance (color, glossiness, texture, liquid-like qualities), and how they're interacting. Include the environment, background, lighting, and atmospheric qualities. This should be a complete description that works independently without referencing the video.

            2. VIDEO DESCRIPTION:
            Describe the complete symbiote transformation sequence shown in the video. Detail how the symbiote substance appears, moves, spreads, and bonds with the subject. Explain the visual progression of the transformation, the changing appearance of both subject and symbiote, and the environmental context throughout. Capture the fluid, alien nature of the symbiote material and the dramatic visual changes it creates. And whether the background is static or dynamic (should be clearly described, if the background remain still, it should be described as static, if the background is dynamic, it should be described as dynamic and describe the detail). This should be a thorough description that works independently without referencing the static image.

            IMPORTANT: DO NOT include any timestamps or time markers in your descriptions. Instead, describe the transformation as a continuous narrative or through its stages.

            Clearly label the two descriptions with "IMAGE:" and "VIDEO:".
            """

            print("Generating descriptions...")
            # Use the most advanced Gemini model
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",  # Using the most advanced model
                contents=[
                    image,
                    video_file,
                    prompt
                ]
            )

            # Delete the uploaded file to save space
            client.files.delete(name=video_file.name)

            return response.text
        except Exception as e:
            print(f"Error in API call (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff with some randomness
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"All {max_retries} attempts failed for {video_path}")
                return None

def extract_descriptions(text):
    """Extract image and video descriptions from the API response text."""
    if not text:
        return None, None

    image_description = None
    video_description = None

    # Find the IMAGE: section
    if "IMAGE:" in text:
        image_part = text.split("IMAGE:")[1]
        if "VIDEO:" in image_part:
            image_description = image_part.split("VIDEO:")[0].strip()
        else:
            image_description = image_part.strip()

    # Find the VIDEO: section
    if "VIDEO:" in text:
        video_description = text.split("VIDEO:")[1].strip()

    return image_description, video_description

def clean_description(text):
    """Clean up the description text to remove unwanted patterns."""
    if not text:
        return ""

    # Remove markdown-style emphasis markers (**text**)
    text = re.sub(r'\*\*', '', text)

    # Remove excessive newlines
    text = re.sub(r'\n\n+', ' ', text)

    # Replace single newlines with spaces
    text = re.sub(r'\n', ' ', text)

    # Remove any extra spaces
    text = re.sub(r' +', ' ', text)

    return text.strip()

def get_relative_path(full_path, base_prefix="/Users/benedict/Downloads"):
    """Get a path relative to the base prefix."""
    if full_path.startswith(base_prefix):
        return full_path[len(base_prefix):].lstrip('/')
    return full_path

def main():
    # Directories
    input_csv = 'datasets/venom/henshin_videos.csv'
    output_dir = 'datasets/venom/all_processed_original_speed'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all video files
    input_videos = []
    df = pd.read_csv(input_csv)
    for index, row in df.iterrows():
        video_folder = row['文件id']
        video_id = str(row['命名']).zfill(3)
        video_file = os.path.join('datasets/venom', video_folder, f'{video_id}.mp4')
        input_videos.append(video_file)
    
    # Process each video
    processed_files = {}

    # Using tqdm for the main processing loop
    for video_file in tqdm(input_videos, desc="Processing videos"):
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(video_file))[0]

        # Add visual separator for each file processing
        print("\n" + "="*80)
        print(f"PROCESSING FILE: {base_name}")
        print("="*80)

        # Extract first frame
        image_output = os.path.join(output_dir, f"{base_name}.jpg")
        if extract_first_frame(video_file, image_output):
            print(f"Extracted first frame from {video_file} to {image_output}")
        else:
            print(f"Failed to extract first frame from {video_file}")
            continue

        # Adjust video speed or copy if /vepfs-zulution/qiufeng/software/ffmpeg-7.0.2-amd64-static/ffmpeg is not available
        video_output = os.path.join(output_dir, f"{base_name}.mp4")
        if adjust_video_speed(video_file, video_output, target_duration=None):
            print(f"Processed video {video_file} to {video_output}")
        else:
            print(f"Failed to process video {video_file}")
            continue

        # Process with Gemini API
        print(f"Processing through Gemini API: {os.path.basename(image_output)} and {os.path.basename(video_output)}")
        descriptions = process_image_video_pair(image_output, video_output)

        if descriptions:
            # Parse the descriptions
            image_description, video_description = extract_descriptions(descriptions)

            # Clean the descriptions and add the fixed opening
            cleaned_image_desc = "" + clean_description(image_description)
            cleaned_video_desc = "Venom symbiote transformation. " + clean_description(video_description)

            # Display only cleaned descriptions
            print("\n----- IMAGE CAPTION -----")
            print(cleaned_image_desc)
            print("\n----- VIDEO CAPTION -----")
            print(cleaned_video_desc)
            print("--------------------------\n")

            processed_files[base_name] = {
                'image_path': os.path.abspath(image_output),
                'video_path': os.path.abspath(video_output),
                'image_description': cleaned_image_desc,
                'video_description': cleaned_video_desc
            }

            print(f"Successfully processed {base_name}")
        else:
            # Add to processed files even if descriptions failed
            default_image_desc = ""
            default_video_desc = "Venom symbiote transformation."

            # Display default captions
            print("\n----- DEFAULT IMAGE CAPTION -----")
            print(default_image_desc)
            print("\n----- DEFAULT VIDEO CAPTION -----")
            print(default_video_desc)
            print("--------------------------------\n")

            processed_files[base_name] = {
                'image_path': os.path.abspath(image_output),
                'video_path': os.path.abspath(video_output),
                'image_description': default_image_desc,
                'video_description': default_video_desc
            }

            print(f"Failed to get descriptions for {base_name}")

        # Optional: Add delay to avoid API rate limits
        time.sleep(2)

    # Generate metadata.json in the exact format requested
    metadata_path = os.path.join(output_dir, 'metadata.json')
    metadata = []

    print("\n" + "="*80)
    print("GENERATING METADATA ENTRIES")
    print("="*80)

    # Process all paired files
    for base_name, info in sorted(processed_files.items()):
        # Add video entry first - using relative path without the /Users/benedict/Downloads prefix
        video_entry = {
            "file_path": get_relative_path(info['video_path']),
            "text": info['video_description'],
            "type": "video"
        }
        metadata.append(video_entry)
        print(f"Added video metadata: {base_name}.mp4")
        print(f"Caption: {video_entry['text']}")

        # Then add matching image entry - using relative path without the /Users/benedict/Downloads prefix
        image_entry = {
            "file_path": get_relative_path(info['image_path']),
            "text": info['image_description'],
            "type": "image"
        }
        metadata.append(image_entry)
        print(f"Added image metadata: {base_name}.jpg")
        print(f"Caption: {image_entry['text']}")

    # Save metadata.json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated metadata file at {metadata_path}")


if __name__ == "__main__":
    main()
