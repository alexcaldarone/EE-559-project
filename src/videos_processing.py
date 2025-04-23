import os
import subprocess
from pathlib import Path


def extract_hate_snippet(video_path: Path, start: float, end: float, output_path: Path):
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
        str(output_path)
    ]
    subprocess.run(cmd, check=True)


def process_video(video_path, annotations, output_dir):
    """
    Creates hate snippets from a video based on the provided annotations.
    
    Parameters:
    - video_path: str, path to the video file
    - annotations: pd.DataFrame, DataFrame containing the annotations
    """

    video_name = os.path.basename(video_path)

    # Get Hateful Snippets
    video_hate_snippets = eval(annotations[annotations['video_file_name'] == video_name]['hate_snippet'].values[0])
    print(video_hate_snippets)
    video_hate_snippets_seconds = [[sum(x*int(t) for x, t in zip([3600, 60, 1], time.split(':'))) for time in hate_snippets] for hate_snippets in video_hate_snippets] # Convert to seconds

    # Check if snippets are less than 5 seconds appart then concatenate, format 00:00:00
    video_hate_snippets_concat = []
    for i in range(len(video_hate_snippets)):
        if i == 0:
            video_hate_snippets_concat.append(video_hate_snippets[i])
        else:
            if video_hate_snippets_seconds[i][0] - video_hate_snippets_seconds[i-1][1] < 5:
                video_hate_snippets_concat[-1][1] = video_hate_snippets[i][1]
            else:
                video_hate_snippets_concat.append(video_hate_snippets[i])
        
    print(video_hate_snippets_concat)
    
    # Create directory to save video snippet as mp4
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract each hate snippet
    for i in range(len(video_hate_snippets)):
        start = video_hate_snippets[i][0]
        end = video_hate_snippets[i][1]
        output_path = os.path.join(output_dir, f"{video_name.split('.')[0]}_snippet_{i}.mp4")
        extract_hate_snippet(video_path, start, end, output_path)