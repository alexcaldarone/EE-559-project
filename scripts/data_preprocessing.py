import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"

logger = setup_logger("data_preprocessing")

def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline")

    parser.add_argument(
        "--steps", nargs="+", default=["all"],
        choices=["all", "snippets", "audio", "text", "frames"],
        help="Which pipeline steps to run"
    )

    parser.add_argument(
        "--model", type=str, default="base",
        help="Whisper model to use (e.g. 'tiny', 'base', 'small', 'medium', 'large')"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files if they already exist"
    )

    return parser.parse_args()

def extract_video_snippets(hateful_videos, non_hateful_videos, annotations, output_dirs, seed):
    for video in tqdm(hateful_videos, desc="Processing hateful videos"):
        process_video(video, annotations, output_dirs['hateful'], seed=seed)
    for video in tqdm(non_hateful_videos, desc="Processing non-hateful videos"):
        process_video(video, annotations, output_dirs['non_hateful'], seed=seed)

def extract_audio_from_snippets(snippets_dir, output_dir, overwrite):
    snippets = list(snippets_dir.glob("*.mp4"))
    for video in tqdm(snippets, desc=f"Extracting audio to {output_dir.name}"):
        audio_path = output_dir / f"{video.stem}.mp3"
        if audio_path.exists() and not overwrite:
            logger.info(f"Skipping {audio_path} (already exists)")
            continue
        extract_audio_from_video(video, audio_path)

def transcribe_audio_to_text(audio_dir, text_dir, model, overwrite):
    audio_files = list(audio_dir.glob("*.mp3"))
    for audio in tqdm(audio_files, desc=f"Transcribing {audio_dir.name}"):
        text_path = text_dir / f"{audio.stem}.txt"
        if text_path.exists() and not overwrite:
            logger.info(f"Skipping {text_path} (already exists)")
            continue
        audio_to_text(audio, text_path, model)

def extract_video_frames(snippets_dir, frames_dir, overwrite):
    video_files = list(snippets_dir.glob("*.mp4"))
    for video in tqdm(video_files, desc=f"Extracting frames from {snippets_dir.name}"):
        frame_output = frames_dir / video.stem
        if frame_output.exists() and not overwrite:
            logger.info(f"Skipping {frame_output} (already exists)")
            continue
        frame_output.mkdir(parents=True, exist_ok=True)
        extract_frames(video, frame_output, fps=1)

if __name__ == "__main__":
    args = parse_args()

    SEED = args.seed
    np.random.seed(SEED)

    steps = args.steps
    if "all" in steps:
        steps = ["snippets", "audio", "text", "frames"]

    annotations = pd.read_csv(RAW_DATA / "HateMM_annotation.csv")

    hateful_raw = RAW_DATA / "videos/hateful/hate_videos"
    non_hateful_raw = RAW_DATA / "videos/non-hateful/non_hate_videos"

    hateful_videos = [hateful_raw / v for v in os.listdir(hateful_raw)]
    non_hateful_videos = [non_hateful_raw / v for v in os.listdir(non_hateful_raw)]

    # dictionary to track paths for saving clean data
    paths = {
        "snippets": {
            "hateful": CLEAN_DATA / "videos/hateful",
            "non_hateful": CLEAN_DATA / "videos/non-hateful",
        },
        "audios": {
            "hateful": CLEAN_DATA / "audios/hateful",
            "non_hateful": CLEAN_DATA / "audios/non-hateful",
        },
        "texts": {
            "hateful": CLEAN_DATA / "texts/hateful",
            "non_hateful": CLEAN_DATA / "texts/non-hateful",
        },
        "frames": {
            "hateful": CLEAN_DATA / "frames/hateful",
            "non_hateful": CLEAN_DATA / "frames/non-hateful",
        },
    }

    # Create directories if they don't exist
    for category in paths.values():
        for p in category.values():
            p.mkdir(parents=True, exist_ok=True)

    if "snippets" in steps:
        extract_video_snippets(hateful_videos, non_hateful_videos, annotations, paths["snippets"], seed=args.seed)

    if "audio" in steps:
        extract_audio_from_snippets(paths["snippets"]["hateful"], paths["audios"]["hateful"], overwrite=args.overwrite)
        extract_audio_from_snippets(paths["snippets"]["non_hateful"], paths["audios"]["non_hateful"], overwrite=args.overwrite)

    if "text" in steps:
        model = whisper.load_model(args.model)
        transcribe_audio_to_text(paths["audios"]["hateful"], paths["texts"]["hateful"], model, overwrite=args.overwrite)
        transcribe_audio_to_text(paths["audios"]["non_hateful"], paths["texts"]["non_hateful"], model, overwrite=args.overwrite)

    if "frames" in steps:
        extract_video_frames(paths["snippets"]["hateful"], paths["frames"]["hateful"], overwrite=args.overwrite)
        extract_video_frames(paths["snippets"]["non_hateful"], paths["frames"]["non_hateful"], overwrite=args.overwrite)