import os
import logging
import warnings

from audio_extract import extract_audio

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: str, audio_path: str):
    """
    Extracts audio from a video file and saves it to the specified output path.

    Parameters:
    - video_path: str, path to the video file
    - audio_path: str, path to save the output audio
    """
    # Check if the output file already exists then skip
    if os.path.exists(audio_path):
        logger.info(f"Audio already extracted from {video_path} to {audio_path}")
        return
    else:
        try:
            extract_audio(video_path, audio_path)
            logger.info(f"Extracted audio from {video_path} to {audio_path}")
        except Exception as e:
            if "does not contain any stream" in str(e):
                logger.error(f"Video {video_path} does not contain any stream. Skipping...")
            else:
                logger.error(f"Error extracting audio from {video_path}: {e}")
                raise # we only skip videos with no stream (make this more robust later)

def audio_to_text(audio_path: str, text_path: str, model):
    """
    Transcribes audio using the Whisper model.

    Parameters:
    - audio_path: str, path to the audio file
    - text_path: str, path to save the transcribed text
    - model_name: str, name of the Whisper model to use (default is "base")
    
    Returns:
    - transcription: str, transcribed text from the audio
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(audio_path)
        text = result["text"]

        with open(text_path, "w") as f:
            f.write(text)
        
        logger.info(f"Transcribed audio {audio_path} to {text_path}")
    except Exception as e:
        logger.error(f"Error transcribing audio {audio_path}: {e}")