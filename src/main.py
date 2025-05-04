from videos_processing import *
from audio2text import *
import pandas as pd
import os


if __name__ == "__main__":  

    annotations = pd.read_csv("../data/raw/HateMM_annotation.csv")

    hateful_videos = [os.path.join("../data/raw/videos/hateful/", video) 
                      for video in os.listdir("../data/raw/videos/hateful/")[:2]]
    hateful_videos_snippets = os.path.join("../data/clean/videos/hateful/")

    non_hateful_videos = [os.path.join("../data/raw/videos/non-hateful/", video) 
                          for video in os.listdir("../data/raw/videos/non-hateful/")[:2]]
    non_hateful_videos_snippets = os.path.join("../data/clean/videos/non-hateful/")

    ### Extract Snippets from Videos
    for video in hateful_videos:
        process_video(video, annotations, hateful_videos_snippets)
        print(f"Processed video: {video}")

    for video in non_hateful_videos:
        process_video(video, annotations, non_hateful_videos_snippets)
        print(f"Processed video: {video}")


    # ### Extract Audio from Videos
    hateful_videos_snippets = [os.path.join(hateful_videos_snippets, video) 
                               for video in os.listdir(hateful_videos_snippets)]
    hateful_videos_audios = os.path.join("../data/clean/audios/hateful/")

    non_hateful_videos_snippets = [os.path.join(non_hateful_videos_snippets, video) 
                                   for video in os.listdir(non_hateful_videos_snippets)]
    non_hateful_videos_audios = os.path.join("../data/clean/audios/non-hateful/")

    for video in hateful_videos_snippets:
        if 'mp4' not in video: continue
        audio_path = os.path.join(hateful_videos_audios, f"{os.path.basename(video).split('.')[0]}.mp3")
        extract_audio_from_video(video, audio_path)
        print(f"Processed audio: {audio_path}")
    
    for video in non_hateful_videos_snippets:
        if 'mp4' not in video: continue
        audio_path = os.path.join(non_hateful_videos_audios, f"{os.path.basename(video).split('.')[0]}.mp3")
        extract_audio_from_video(video, audio_path)
        print(f"Processed audio: {audio_path}")

    
    ### Transcribe Audio to Text
    hateful_audios = [os.path.join(hateful_videos_audios, audio) for audio in os.listdir(hateful_videos_audios)]
    hateful_videos_texts = os.path.join("../data/clean/texts/hateful/")

    non_hateful_audios = [os.path.join(non_hateful_videos_audios, audio) for audio in os.listdir(non_hateful_videos_audios)]
    non_hateful_videos_texts = os.path.join("../data/clean/texts/non-hateful/")

    for audio in hateful_audios:
        text_path = os.path.join(hateful_videos_texts, f"{os.path.basename(audio).split('.')[0]}.txt")
        process_audio(audio_path, text_path)
        print(f"Processed text: {text_path}")
    
    for audio in non_hateful_audios:
        text_path = os.path.join(non_hateful_videos_texts, f"{os.path.basename(audio).split('.')[0]}.txt")
        process_audio(audio_path, text_path)
        print(f"Processed text: {text_path}")

    ### Extract Frames from Snippets

    hateful_frames = os.path.join("../data/clean/frames/hateful/")
    non_hateful_frames = os.path.join("../data/clean/frames/non-hateful/")

    for video in hateful_videos_snippets:
        if 'mp4' not in video: continue
        frames_path = os.path.join(hateful_frames, f"{os.path.basename(video).split('.')[0]}/")
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
        extract_frames(video, frames_path, fps=1)
        print(f"Processed frames: {frames_path}")
    
    for video in non_hateful_videos_snippets:
        if 'mp4' not in video: continue
        frames_path = os.path.join(non_hateful_frames, f"{os.path.basename(video).split('.')[0]}/")
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
        extract_frames(video, frames_path, fps=1)
        print(f"Processed frames: {frames_path}")