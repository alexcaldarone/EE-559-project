import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import whisper

from src.data_processing.videos_processing import process_video, extract_frames
from src.data_processing.audios_processing import extract_audio_from_video, audio_to_text
from src.utils.logger import setup_logger

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

setup_logger("data_preprocessing")

if __name__ == "__main__":
    annotations = pd.read_csv(PROJECT_ROOT / "data/raw/HateMM_annotation.csv")

    hateful_videos = [os.path.join(PROJECT_ROOT / "data/raw/videos/hateful/hate_videos", video) 
                      for video in os.listdir(PROJECT_ROOT / "data/raw/videos/hateful/hate_videos/")]
    hateful_videos_snippets = os.path.join(PROJECT_ROOT / "data/clean/videos/hateful/")

    non_hateful_videos = [os.path.join(PROJECT_ROOT / "data/raw/videos/non-hateful/non_hate_videos", video) 
                          for video in os.listdir(PROJECT_ROOT / "data/raw/videos/non-hateful/non_hate_videos")]
    non_hateful_videos_snippets = os.path.join(PROJECT_ROOT / "data/clean/videos/non-hateful/")

    ### Extract Snippets from Videos
    
    for video in tqdm(hateful_videos, desc="Processing hateful videos (hateful snippet extraction)"):
        process_video(video, annotations, hateful_videos_snippets, seed=SEED)

    for video in tqdm(non_hateful_videos, desc = "Processing non-hateful videos (snippet extraction)"):
        process_video(video, annotations, non_hateful_videos_snippets, seed=SEED)

    # ### Extract Audio from Videos
    hateful_videos_snippets = [os.path.join(hateful_videos_snippets, video) 
                               for video in os.listdir(hateful_videos_snippets)]
    hateful_videos_audios = os.path.join(PROJECT_ROOT / "data/clean/audios/hateful/")

    non_hateful_videos_snippets = [os.path.join(non_hateful_videos_snippets, video) 
                                   for video in os.listdir(non_hateful_videos_snippets)]
    non_hateful_videos_audios = os.path.join(PROJECT_ROOT / "data/clean/audios/non-hateful/")

    for video in tqdm(hateful_videos_snippets, desc="Extracting audio from hateful videos"):
        if 'mp4' not in video: continue
        audio_path = os.path.join(hateful_videos_audios, f"{os.path.basename(video).split('.')[0]}.mp3")
        extract_audio_from_video(video, audio_path)

    for video in tqdm(non_hateful_videos_snippets, desc="Extracting audio from non-hateful videos"):
        if 'mp4' not in video: continue
        audio_path = os.path.join(non_hateful_videos_audios, f"{os.path.basename(video).split('.')[0]}.mp3")
        extract_audio_from_video(video, audio_path)

    ### Transcribe Audio to Text
    hateful_audios = [os.path.join(hateful_videos_audios, audio) for audio in os.listdir(hateful_videos_audios)]
    hateful_videos_texts = os.path.join(PROJECT_ROOT / "data/clean/texts/hateful/")

    non_hateful_audios = [os.path.join(non_hateful_videos_audios, audio) for audio in os.listdir(non_hateful_videos_audios)]
    non_hateful_videos_texts = os.path.join(PROJECT_ROOT / "data/clean/texts/non-hateful/")

    model = whisper.load_model("base")

    for audio in tqdm(hateful_audios, desc="Transcribing hateful audios"):
        text_path = os.path.join(hateful_videos_texts, f"{os.path.basename(audio).split('.')[0]}.txt")
        audio_to_text(audio, text_path, model)

    for audio in tqdm(non_hateful_audios, desc="Transcribing non-hateful audios"):
        text_path = os.path.join(non_hateful_videos_texts, f"{os.path.basename(audio).split('.')[0]}.txt")
        audio_to_text(audio, text_path, model)

    ### Extract Frames from Snippets
    hateful_frames = os.path.join(PROJECT_ROOT / "data/clean/frames/hateful/")
    non_hateful_frames = os.path.join(PROJECT_ROOT / "data/clean/frames/non-hateful/")

    for video in tqdm(hateful_videos_snippets, desc="Extracting frames from hateful videos"):
        if 'mp4' not in video: continue
        frames_path = os.path.join(hateful_frames, f"{os.path.basename(video).split('.')[0]}/")
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
        extract_frames(video, frames_path, fps=1)
    
    for video in tqdm(non_hateful_videos_snippets, desc="Extracting frames from non-hateful videos"):
        if 'mp4' not in video: continue
        frames_path = os.path.join(non_hateful_frames, f"{os.path.basename(video).split('.')[0]}/")
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
        extract_frames(video, frames_path, fps=1)