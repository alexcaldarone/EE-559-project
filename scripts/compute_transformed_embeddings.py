import os
from pathlib import Path
from PIL import Image
import pickle
import yaml
import re
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from src.utils.logger import setup_logger




# Default data directory - replace with your actual path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "data_augmentation.yml"
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"
KEYWORDS_DATA = PROJECT_ROOT / "data/flagged_words.csv"
TRANSFORMED_EMBEDDING_DIR = PROJECT_ROOT / "data/embeddings_transformed"

logger = setup_logger("precompute_transformed_embeddings")

def load_config(config_path):
    """Load configuration from a YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def set_seed(seed: int):
    """Set global seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def augment_text(text_chunks, keywords=None):
    """Augment text chunk to mask out some words.
    
    Args:
            text_chunks: List of text chunks to augment
            keywords: List of keywords to mask out, if `None`, no augmentation is done
    """
    if keywords is None:
        return text_chunks
    # Sort by length to match longer phrases first (avoids partial masking)
    sorted_phrases = sorted(keywords, key=len, reverse=True)
    
    # Create a regex that matches any of the phrases, escaped for safety
    pattern = re.compile(
        r'(' + '|'.join(re.escape(p) for p in sorted_phrases) + r')',
        flags=re.IGNORECASE
    )

    return [pattern.sub("*****", text) for text in text_chunks]

def augment_frames(frames, configs=None):
    """Augment frames to robustify the model
    
        Args:
            frames: List of frames to augment
            configs: Augmentation configurations, if `None`, no augmentation is done
    """
    if configs is None:
        return frames
    
    augmented_frames = []
    
    # Define augmentation transformations
    transform = transforms.Compose([
        transforms.RandomCrop(
            size = configs["RandomCrop"]["size"],
            padding = configs["RandomCrop"]["padding"],
            pad_if_needed= True # Pad the image if smaller than the crop size
        ),
        transforms.RandomHorizontalFlip(
            p = configs["RandomHorizontalFlip"]["p"]
        ),
        transforms.ColorJitter(
            brightness = configs["ColorJitter"]["brightness"],
            contrast = configs["ColorJitter"]["contrast"],
            saturation = configs["ColorJitter"]["saturation"],
            hue = configs["ColorJitter"]["hue"]
        ),
        transforms.RandomRotation(
            degrees = configs["RandomRotation"]["degrees"]
        ),
        transforms.ToTensor()
    ])
    
    for frame in frames:
        try:
            augmented_frame = transform(frame)
            augmented_frames.append(augmented_frame)
        except Exception as e:
            logger.warning(f"Error augmenting frame: {str(e)}")
            continue
            
    return augmented_frames

def show_images(images, cols=5, figsize=(15, 5), title=None, file_name="augmented_frames.pdf"):
    """
    Display a list of PIL Images or torch tensors.
    
    Args:
        images: list of PIL.Image or torch.Tensor (C,H,W) in [0,1]
        cols: number of columns in the grid
    """
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL for consistent display
            img = TF.to_pil_image(img)
        plt.imshow(img)
        if i == 1 and title:
            plt.title(title)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(str(TRANSFORMED_EMBEDDING_DIR) + "/" + file_name)

def process_batch(model, processor, frames, text_chunks, device):
    """Process a batch of frames and text through CLIP"""
    image_embeddings = []
    text_embedding = None
    
    # Process frames in batches
    for frame in frames:
        try:
            inputs = processor(
                text=text_chunks,
                images=frame,
                return_tensors="pt",
                padding=True,
                do_rescale=False # Since the augmentation part already rescales the images
            ).to(device)
            
            with torch.no_grad():
                output = model(**inputs)
                
            image_embeddings.append(output.image_embeds)
            
            # Only need to compute text embedding once
            if text_embedding is None:
                if output.text_embeds.size(0) > 0:
                    text_embedding = output.text_embeds.mean(dim=0, keepdim=True)
                else:
                    text_embedding = torch.zeros_like(output.text_embeds[0]).unsqueeze(0)
                
        except Exception as e:
            logger.warning(f"Error processing frame: {str(e)}")
            continue
    
    if image_embeddings:
        image_embeddings = torch.stack(image_embeddings)
        return image_embeddings, text_embedding
    else:
        return None, None

def save_embeddings(video_id, image_embeddings, text_embedding, label, output_dir):
    """Save embeddings to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'video_id': video_id,
        'image_embeddings': image_embeddings.cpu(),
        'text_embedding': text_embedding.cpu(),
        'label': label
    }
    
    output_file = os.path.join(output_dir, f"{video_id}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    return output_file

def main():
    # Load configuration
    configs = load_config(CONFIG_FILE)
    set_seed(configs["seed"])
    logger.info(f"Using seed: {configs['seed']}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(TRANSFORMED_EMBEDDING_DIR, exist_ok=True)
    
    # Load CLIP model
    logger.info(f"Loading CLIP model on {device}...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info(f"CLIP model loaded succesfully.")
    except ImportError as e:
        logger.info(f"Failed loading CLIP model with error: {e}")
        raise ImportError
    
    # Get all video IDs
    all_video_ids = get_video_ids(CLEAN_DATA)
    logger.info(f"Found {len(all_video_ids)} videos")
    
    # Validate video IDs
    valid_video_ids = validate_video_ids(all_video_ids, CLEAN_DATA, logger)
    logger.info(f"Validated {len(valid_video_ids)} videos with both frames and text")
    
    # Create an index file to map video_ids to labels
    index_data = []

    # Process each video
    for video_id in tqdm(valid_video_ids, desc="Processing videos"):
        # Determine label
        video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
        label = 0 if video_label == 'non-hateful' else 1
        
        # Skip if embeddings already exist
        output_file = os.path.join(TRANSFORMED_EMBEDDING_DIR, f"{video_id}.pkl")
        if os.path.exists(output_file):
            logger.info(f"Embeddings for {video_id} already exist, skipping")
            index_data.append({'video_id': video_id, 'label': label, 'file': output_file})
            continue

        # Load text
        text_chunks = load_text(video_id, CLEAN_DATA, logger)
        if not text_chunks:
            logger.warning(f"No valid text for {video_id}, skipping")
            continue

        # Load frames
        frames = load_frames(video_id, CLEAN_DATA, logger)
        if not frames:
            logger.warning(f"No valid frames for {video_id}, skipping")
            continue

        # Show original images
        #show_images(frames[:5], title = video_id + " - Original Frames", file_name = video_id + "_original.pdf")
        
        # Augment text and frames
        if configs["text_augmentation"]["apply"]:
            keywords = pd.read_csv(KEYWORDS_DATA, header=None)
            keywords = keywords.loc[:, 0].tolist()
            text_chunks = augment_text(text_chunks, keywords)
        if configs["frame_augmentation"]["apply"]:
            frames = augment_frames(frames, configs["frame_augmentation"]["augmentations"])

        # Show augmented images
        #show_images(frames[:1], title = video_id + " - Augmented Frames", file_name = video_id + "_augmented.pdf")

        # Process through CLIP
        image_embeddings, text_embedding = process_batch(
            model, processor, frames, text_chunks, device
        )
        
        if image_embeddings is None or text_embedding is None:
            logger.warning(f"Failed to generate embeddings for {video_id}, skipping")
            continue
            
        # Save embeddings
        saved_file = save_embeddings(
            video_id, image_embeddings, text_embedding, label, TRANSFORMED_EMBEDDING_DIR
        )
        
        logger.info(f"Saved embeddings for {video_id} to {saved_file}")
        index_data.append({'video_id': video_id, 'label': label, 'file': saved_file})
    
    # Save index file
    index_file = os.path.join(TRANSFORMED_EMBEDDING_DIR, "index.pkl")
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)
    
    logger.info(f"Embeddings saved for {len(index_data)} videos. Index saved to {index_file}")


if __name__ == "__main__":
    main()