from videos_processing import *
from audio2text import *
import pandas as pd
import os


if __name__ == "__main__":  

    annotations = pd.read_csv("../data/raw/HateMM_annotation.csv")

    hateful_videos = os.listdir("../data/raw/videos/hateful/")[:3] # For testing purposes
    hateful_videos = [os.path.join("../data/raw/videos/hateful/", video) for video in hateful_videos]
    hateful_videos_snippets = os.path.join("../data/clean/videos/hateful/")

    # Extract Hateful Snippets from Videos
    for video in hateful_videos:
        process_video(video, annotations, hateful_videos_snippets)
        print(f"Processed video: {video}")

    # Extract Audio from Videos
    hateful_videos_audios = os.path.join("../data/clean/audios/hateful/")

    for video in hateful_videos:
        audio_path = os.path.join(hateful_videos_audios, f"{os.path.basename(video).split('.')[0]}.mp3")
        extract_audio_from_video(video, audio_path)
        print(f"Processed audio: {audio_path}")
    
    # Transcribe Audio to Text
    hateful_videos_texts = os.path.join("../data/clean/texts/hateful/")
    hateful_audios = os.listdir(hateful_videos_audios)
    hateful_audios = [os.path.join(hateful_videos_audios, audio) for audio in hateful_audios]

    for audio in hateful_audios:
        text_path = os.path.join(hateful_videos_texts, f"{os.path.basename(audio).split('.')[0]}.txt")
        process_audio(audio_path, text_path)
        print(f"Processed text: {text_path}")
        