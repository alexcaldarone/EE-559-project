import os
import subprocess
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def get_length(filename: str):
    """
    Get the length of a video file in seconds.
    Source: https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
    """

    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def extract_snippet(video_path: Path, start: float, end: float, output_path: Path):
    """
    Extracts a snippet from a video and saves it to the specified output path.

    !!! NOTE: Needs to have ffmpeg installed on the machine.
    
    Parameters:
    - video_path: str, path to the video file
    - start: float, start time in seconds
    - end: float, end time in seconds
    - output_path: str, path to save the output video
    """
    cmd = [
        "ffmpeg",
        "-ss", start,
        "-to", end,
        "-i", str(video_path),
        "-c:v", "copy",
        "-c:a", "copy",
        "-y",
        "-loglevel", "error",
        "-hide_banner",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Extracted snippet {[start, end]} from {video_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        # write failure to logs but do not raise an error and interrupt the pipeline
        logger.error(f"Error extracting snippet {[start, end]} from {video_path}: {e}")


def process_video(video_path: str, annotations: pd.DataFrame, output_dir: str, seed=None, min_time=5):
    """
    Creates hate snippets from a video based on the provided annotations.
    
    Parameters:
    - video_path: str, path to the video file
    - annotations: pd.DataFrame, DataFrame containing the annotations
    - output_dir: str, directory to save the output video snippets
    - min_time: int, minimum time in seconds to consider a snippet
    """
    logger.info("Processing video: %s", video_path)

    try:
        if seed is not None:
            np.random.seed(seed)

        video_name = os.path.basename(video_path)

        if 'non_hate' in video_name:
            # Get Video Length
            video_length = get_length(video_path)

            # Get a Random Number of Snippets
            num_snippets = np.random.randint(1, 3)

            # Get Random Non-Hateful Snippets
            video_snippets = []

            if video_length <= min_time:
                logger.warning(f"Video {video_name} is too short for random snippets. Taking whole video.")
                video_snippets.append([0, video_length])
            else:
                for _ in range(num_snippets):
                    start = np.random.randint(0, video_length - min_time)
                    end   = np.random.randint(start + min_time, video_length + 1)
                    video_snippets.append([start, end])

            # Convert seconds to time format
            video_snippets = [[f"{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02}",
                            f"{int(end//3600):02}:{int((end%3600)//60):02}:{int(end%60):02}"] for start, end in video_snippets]

        else:
            # Get Hateful Snippets
            video_snippets = eval(annotations[annotations['video_file_name'] == video_name]['hate_snippet'].values[0])
            video_snippets_seconds = [[sum(x*int(t) for x, t in zip([3600, 60, 1], time.split(':'))) for time in sorted(hate_snippets)] 
                                    for hate_snippets in sorted(video_snippets)] # Convert to seconds
            
            # Check if snippets are less than 5 seconds appart then concatenate
            video_snippets_concat = []
            for i in range(len(video_snippets)):
                if i == 0:
                    video_snippets_concat.append(video_snippets[i])
                else:
                    if video_snippets_seconds[i][0] - video_snippets_seconds[i-1][1] < 5:
                        video_snippets_concat[-1][1] = video_snippets[i][1]
                    else:
                        video_snippets_concat.append(video_snippets[i])
            video_snippets = video_snippets_concat
            
        # Create directory to save video snippet as mp4
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract each snippet
        for i in range(len(video_snippets)): # se qui uso seconds e poi riconverto in stringa di tempo?
            start = video_snippets[i][0]
            end = video_snippets[i][1]
            if start > end: # lexographic comparison works because it's in hh:mm:ss format
                start, end = end, start # flip them back
            output_path = os.path.join(output_dir, f"{video_name.split('.')[0]}_snippet_{i}.mp4")
            extract_snippet(video_path, start, end, output_path)
        
        logger.info(f"Successfully extracted {len(video_snippets)} snippet(s) from {video_name}")
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")


def extract_frames(input_path: str, output_dir: str, fps: int = 1):
    """
    Extracts frames from a video file and saves them as JPEG images.

    Parameters:
    - input_path: str, path to the input video file
    - output_dir: str, directory to save the extracted frames
    - fps: int, frames per second to extract (default is 1)
    """

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", input_path,         
        "-vf", f"fps={fps}",
        "-q:v", "2",              # output quality (for JPEG 2)
        os.path.join(output_dir, "frame_%04d.jpg"),
        "-loglevel", "error",
        "-hide_banner",
    ]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Extracted frames from {input_path} to {output_dir}")
    except subprocess.CalledProcessError as e:
        # write failure to logs but do not raise an error and interrupt the pipeline
        logger.error(f"Error extracting frames from {input_path}: {e}")