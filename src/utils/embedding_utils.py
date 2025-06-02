import os
from PIL import Image

def get_video_ids(root_dir):
    """Get all available video IDs by checking both hateful and non-hateful directories"""
    video_ids = []
    
    for label in ['hateful', 'non-hateful']:
        frames_dir = os.path.join(root_dir, f'frames/{label}')
        if os.path.exists(frames_dir):
            video_ids.extend([vid for vid in os.listdir(frames_dir) 
                             if os.path.isdir(os.path.join(frames_dir, vid))])
    
    return video_ids

def validate_video_ids(video_ids, root_dir, logger):
    """Validate video IDs and return only those with valid frames and text"""
    valid_ids = []
    
    for video_id in video_ids:
        try:
            video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
            
            # Check if video folder exists
            video_folder = os.path.join(root_dir, f'frames/{video_label}/{video_id}/')
            
            # Check if text file exists
            text_file = os.path.join(root_dir, f'texts/{video_label}/{video_id}.txt')
            
            if not os.path.exists(video_folder):
                logger.warning(f"Skipping {video_id}: folder not found: {video_folder}")
                continue
                
            frame_files = [f for f in os.listdir(video_folder) 
                          if f.endswith((".jpg", ".png"))]
            if not frame_files:
                logger.warning(f"Skipping {video_id}: no frame files in {video_folder}")
                continue
                
            if not os.path.exists(text_file):
                logger.warning(f"Skipping {video_id}: text file not found: {text_file}")
                continue
                
            # All checks passed, this is a valid video ID
            valid_ids.append(video_id)
            
        except Exception as e:
            logger.error(f"Error validating {video_id}: {str(e)}")
            continue
            
    return valid_ids

def load_text(video_id, root_dir, logger):
    """Load and process text for a video"""
    video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
    text_file = os.path.join(root_dir, f'texts/{video_label}/{video_id}.txt')
    
    try:
        with open(text_file, 'r') as f:
            text = f.read().strip()

        if not text:
            logger.warning(f"Warning: Empty text for {video_id}, using placeholder.")
            text = "empty"
    except Exception as e:
        logger.error(f"Error loading text for {video_id}: {str(e)}")
        text = "empty"
    
    # Chunk the text into blocks of 30 words
    text = text.split()
    text_chunks = [' '.join(text[i:i + min(30, len(text))]) for i in range(0, len(text), 30)]
    
    return text_chunks

def load_frames(video_id, root_dir, logger):
    """Load frames for a video"""
    video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
    video_folder = os.path.join(root_dir, f'frames/{video_label}/{video_id}/')
    
    try:
        frame_files = sorted([
            os.path.join(video_folder, fname)
            for fname in os.listdir(video_folder)
            if fname.endswith((".jpg", ".png"))
        ])

        if not frame_files:
            raise FileNotFoundError(f"No frame files found for {video_id}")
            
        frames = []
        for f in frame_files:
            try:
                img = Image.open(f).convert("RGB")
                frames.append(img)
            except Exception as e:
                logger.warning(f"Error loading image {f}: {str(e)}")
                continue
                
        return frames
    except Exception as e:
        logger.error(f"Error loading frames for {video_id}: {str(e)}")
        return []