import argparse
import os
import random
from pathlib import Path
from datetime import datetime
import pickle

import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.model.dataset import video_batcher
from src.utils.logger import setup_logger

logger = setup_logger("finetune_clip")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"
FINETUNED_EMBEDDINGS_DIR = PROJECT_ROOT / "data/finetuned_embeddings"
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "models/checkpoints/clip_finetune"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# from scripts/compute_embeddings.py (move to utils)
def get_video_ids(root_dir):
    """Get all available video IDs by checking both hateful and non-hateful directories"""
    video_ids = []
    
    for label in ['hateful', 'non-hateful']:
        frames_dir = os.path.join(root_dir, f'frames/{label}')
        if os.path.exists(frames_dir):
            video_ids.extend([vid for vid in os.listdir(frames_dir) 
                             if os.path.isdir(os.path.join(frames_dir, vid))])
    
    return video_ids

# from scripts/compute_embeddings.py (move to utils)
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

class RawVideoDataset(Dataset):
    """
    Dataset that returns raw frames and transcript for each video.
    Assumes directory structure:
      frames_root/hateful/<video_id>/*.jpg
      frames_root/non-hateful/<video_id>/*.jpg
      transcripts_root/hateful/<video_id>.txt
      transcripts_root/non-hateful/<video_id>.txt

    Samples up to `num_frames` per video uniformly.
    """
    def __init__(
            self, 
            root_dir: Path,
            clean_data_dir: Path,
            frames_root: Path, 
            transcripts_root: Path,
            valid_ids: list,
            processor: CLIPProcessor,
            logger
        ):
        self.ids = valid_ids
        self.root_dir = Path(root_dir)
        self.clean_data_dir = Path(clean_data_dir)
        self.frames_root = Path(frames_root)
        self.transcripts_root = Path(transcripts_root)
        self.processor = processor
        self.logger = logger


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        # Determine label by folder membership
        video_label = 'non-hateful' if 'non_hate' in vid else 'hateful'
        label = 1 if video_label == 'hateful' else 0

        images = self.__load_frames(vid)
        text = self.__load_text(vid)

        return images, text, label, vid
    
    def __load_text(self, video_id):
        """Load and process text for a video"""
        video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
        text_file = os.path.join(self.clean_data_dir, f'texts/{video_label}/{video_id}.txt')
        
        try:
            with open(text_file, 'r') as f:
                text = f.read().strip()

            if not text:
                self.logger.warning(f"Warning: Empty text for {video_id}, using placeholder.")
                text = "empty"
        except Exception as e:
            self.logger.error(f"Error loading text for {video_id}: {str(e)}")
            text = "empty"
        
        # Chunk the text into blocks of 30 words
        text = text.split()
        text_chunks = [' '.join(text[i:i + min(30, len(text))]) for i in range(0, len(text), 30)]
        
        return text_chunks

    def __load_frames(self, video_id):
        """Load frames for a video"""
        video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
        video_folder = os.path.join(self.clean_data_dir, f'frames/{video_label}/{video_id}/')
        
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
                    self.logger.warning(f"Error loading image {f}: {str(e)}")
                    continue
                    
            return frames
        except Exception as e:
            self.logger.error(f"Error loading frames for {video_id}: {str(e)}")
            return []


def collate_fn(batch, processor):
    images_list, text_chunks_list, labels, vids = zip(*batch)
    #print(images_list)
    # Flatten images to list
    flat_images = [img for images in images_list for img in images]
    flat_texts = [" ".join(chunks) for chunks in text_chunks_list]
    # Process via CLIPProcessor
    
    encodings_list = []
    for frame in images_list[0]:
        #print(frame)
        #print(text_chunks_list, type(text_chunks_list))
        encoded = processor(
            text=text_chunks_list[0], 
            images=frame,
            return_tensors='pt', 
            padding=True)
        encodings_list.append(encoded)
    #print("encoded all frames successfully")

    # We need to keep track of offsets to recombine per-video
    num_frames = [len(images) for images in images_list]
    return encodings_list, torch.tensor(labels), vids, num_frames


def build_classification_head(embed_dim: int):
    """Classification head on top of CLIP pooled embeddings."""
    return nn.Sequential(
        nn.Linear(2 * embed_dim, embed_dim),
        nn.ReLU(),
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 2)
    )

def freeze_all_but_projections(model):
    for name, param in model.named_parameters():
        if not ("vision_projection" in name or "text_projection" in name):
            param.requires_grad = False

@torch.no_grad()
def evaluate(model, head, loader, criterion, device):
    model.eval(); head.eval()
    total_loss = 0
    total_samples = 0

    for encoded_list, labels, _, num_frames in loader:
        for encoded in encoded_list:
            for k, v in encoded.items():
                encoded[k] = v.to(device)
        labels = labels.to(device)

        try:
            outputs = model(**encoded)
        except:
            continue

        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds

        pooled_imgs = []
        idx = 0
        for nf in num_frames:
            pooled_imgs.append(img_embeds[idx:idx+nf].mean(dim=0))
            idx += nf
        pooled_imgs = torch.stack(pooled_imgs, dim=0)

        txt_embeds = txt_embeds.mean(dim=0, keepdim=True)
        fused = torch.cat([pooled_imgs, txt_embeds], dim=1)

        logits = head(fused)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples

def finetune_clip(config, device):
    # Load CLIP
    model = CLIPModel.from_pretrained(config['model']).to(device)
    processor = CLIPProcessor.from_pretrained(config['model'])
    # Freeze nothing: fine-tune both encoders

    freeze_all_but_projections(model)

    # Attach classification head
    clip_dim = model.config.projection_dim
    head = build_classification_head(clip_dim).to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config['lr']),
        weight_decay=float(config['weight_decay'])
    )
    criterion = nn.CrossEntropyLoss()

    # Prepare data splits - change this to use global vars
    video_ids = get_video_ids(CLEAN_DATA)
    valid_video_ids = validate_video_ids(video_ids, CLEAN_DATA, logger)
    random.shuffle(valid_video_ids)
    split = int(len(valid_video_ids) * config['train_split'])
    train_ids, test_ids = valid_video_ids[:split], valid_video_ids[split:]
    logger.info(f"There are {len(valid_video_ids)} valid ids")

    train_dataset = RawVideoDataset(
        root_dir=PROJECT_ROOT,
        clean_data_dir=CLEAN_DATA,
        frames_root=CLEAN_DATA/"frames",
        transcripts_root=CLEAN_DATA/"texts",
        valid_ids=train_ids,
        processor=processor,
        logger=logger
    )

    test_dataset = RawVideoDataset(
        root_dir=PROJECT_ROOT,
        clean_data_dir=CLEAN_DATA,
        frames_root=CLEAN_DATA/"frames",
        transcripts_root=CLEAN_DATA/"texts",
        valid_ids=test_ids,
        processor=processor,
        logger=logger
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    test_loader   = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    writer = SummaryWriter(PROJECT_ROOT / "logs")
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(config['epochs']):
        model.train(); head.train()
        for encoded_list, labels, vids, num_frames in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{config['epochs']}"):
            
            for encoded in encoded_list:
                for k, v in encoded.items():
                    encoded[k] = v.to(device)
                labels = labels.to(device)

                # catches when there are too many tokens (often sign input is problematic)
                try:
                    outputs = model(**encoded)
                except Exception as e:
                    logger.warning(
                        f"Skipping batch for videos {vids} at epoch {epoch+1} "
                        f"due to error in forward pass: {e}"
                    )
                    continue
                img_embeds = outputs.image_embeds  # (sum_frames, dim)
                txt_embeds = outputs.text_embeds   # (batch, dim)

                # Pool image embeds per video
                pooled_imgs = []
                idx = 0
                for nf in num_frames:
                    pooled = img_embeds[idx:idx+nf].mean(dim=0)
                    pooled_imgs.append(pooled)
                    idx += nf
                pooled_imgs = torch.stack(pooled_imgs, dim=0)

                # Classification
                txt_embeds = txt_embeds.mean(dim=0, keepdim=True)
                fused = torch.cat([pooled_imgs, txt_embeds], dim=1)
                logits = head(fused)
                loss = criterion(logits, labels)

                optimizer.zero_grad(); loss.backward(); optimizer.step()

                writer.add_scalar("Loss/train", loss.item(), global_step)
                #logger.info(f"Step {global_step} | Epoch {epoch+1} | Loss={loss.item():.4f}")
                global_step += 1

        val_loss = evaluate(model, head, test_loader, criterion, device)
        writer.add_scalar("Loss/val", val_loss, epoch)
        logger.info(f"Epoch {epoch+1} | Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
            torch.save(head.state_dict(), MODEL_CHECKPOINTS_DIR / 'best_head.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        writer.close()

    # Save fine-tuned CLIP and head
    os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
    model.save_pretrained(MODEL_CHECKPOINTS_DIR/ 'clip-finetuned')
    processor.save_pretrained(MODEL_CHECKPOINTS_DIR / 'clip-finetuned')
    head_path = MODEL_CHECKPOINTS_DIR / 'classification_head.pt'
    torch.save(head.state_dict(), head_path)

# replace with process batch
def extract_embeddings(config, device):
    # Load fine-tuned CLIP
    model = CLIPModel.from_pretrained(MODEL_CHECKPOINTS_DIR / 'clip-finetuned').to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_CHECKPOINTS_DIR / 'clip-finetuned')
    model.eval()

    index_data = []

    video_ids = get_video_ids(CLEAN_DATA)
    valid_video_ids = validate_video_ids(video_ids, CLEAN_DATA, logger)
    random.shuffle(valid_video_ids)

    os.makedirs(FINETUNED_EMBEDDINGS_DIR, exist_ok=True)
    
    # dataset to iterate over all the valid id's to get and save embeddings
    dataset = RawVideoDataset(
        root_dir=PROJECT_ROOT,
        clean_data_dir=CLEAN_DATA,
        frames_root=CLEAN_DATA/"frames",
        transcripts_root=CLEAN_DATA/"texts",
        valid_ids=valid_video_ids,
        processor=processor,
        logger=logger
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    for encoded_list, labels, vids, _ in tqdm(loader, desc="Computing finetuned embeddings"):
            for encoded in encoded_list:
                for k, v in encoded.items():
                    encoded[k] = v.to(device)
                labels = labels.to(device)

                image_embeddings = []
                text_embedding = None
                # catches when there are too many tokens (often sign input is problematic)
                try:
                    outputs = model(**encoded)

                    image_embeddings.append(outputs.image_embeds)

                    if text_embedding is None:
                        if outputs.text_embeds.size(0) > 0:
                            text_embedding = outputs.text_embeds.mean(dim=0, keepdim=True)
                        else:
                            text_embedding = torch.zeros_like(outputs.text_embeds[0]).unsqueeze(0)
                            
                except Exception as e:
                    logger.warning(
                        f"Skipping batch for videos {vids}"
                        f"due to error in forward pass: {e}"
                    )
                    continue
                    
            
            if image_embeddings:
                image_embeddings = torch.stack(image_embeddings)
            if isinstance(image_embeddings, list) and len(image_embeddings) == 0:
                continue
            #print(type(image_embeddings))
            #print(image_embeddings)
            data = {
                'video_id': vids[0],
                'image_embeddings': image_embeddings.cpu(),
                'text_embedding': text_embedding.cpu(),
                'label': labels.cpu()
            }

            output_file = os.path.join(FINETUNED_EMBEDDINGS_DIR, f"{vids[0]}.pkl")
            index_data.append({'video_id': vids[0], 'label': labels, 'file': output_file})
            with open(output_file, "wb") as f:
                pickle.dump(data, f)
    
    index_file = os.path.join(FINETUNED_EMBEDDINGS_DIR, "index.pkl")
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)


def main():
    parser = argparse.ArgumentParser("Finetune CLIP and extract embeddings")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(config["seed"])
    # Phase 1: Finetune CLIP
    #finetune_clip(config, device)
    # Phase 2: Extract embeddings with finetuned weights
    extract_embeddings(config, device)


if __name__ == '__main__':
    main()