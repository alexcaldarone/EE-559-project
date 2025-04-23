from audio_extract import extract_audio
import whisper
import os

def extract_audio_from_video(video_path: str, audio_path: str):
    """
    Extracts audio from a video file and saves it to the specified output path.

    Parameters:
    - video_path: str, path to the video file
    - audio_path: str, path to save the output audio
    """
    # Check if the output file already exists then skip
    if os.path.exists(audio_path):
        print(f"Audio already extracted from {video_path} to {audio_path}")
        return
    else:
        extract_audio(video_path, audio_path)

def process_audio(audio_path: str, text_path: str, model_name: str = "base"):
    """
    Transcribes audio using the Whisper model.

    Parameters:
    - audio_path: str, path to the audio file
    - text_path: str, path to save the transcribed text
    - model_name: str, name of the Whisper model to use (default is "base")
    
    Returns:
    - transcription: str, transcribed text from the audio
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    text = result["text"]

    with open(text_path, "w") as f:
        f.write(text)


    