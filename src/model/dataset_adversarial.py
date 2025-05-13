from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DATA = PROJECT_ROOT / "data/embeddings"

class RandomAdversarialVideoTextDataset(Dataset):
    def __init__(
            self, 
            video_ids,
            logger,
            device,
            embedding_dir = EMBEDDINGS_DATA,
            shuffle_text = True, # True if shuffle hateful texts into non-hateful texts, 
                              # False to shuffle videos instead
        ):
        """
        video_ids: unique identifiers for each video e.g. 'hate_video_1_snippet_0'
        embedding_dir: directory where video frame folders are stored (by default ../data/clean)
        shuffle_text: True if shuffle hateful texts into non-hateful texts, 
                      False to shuffle videos instead
        """
        self.video_ids = video_ids
        self.embedding_dir = str(embedding_dir)

        self.logger = logger
        self.device = device

        self.valid_video_ids = self._validate_video_ids()
        self.empty_text_placeholder = "empty"

    def _validate_video_ids(self):
        """
        Pre-check all video IDs to filter out ones with missing files.
        Returns list of valid video IDs.
        """
        valid_ids = []
        for video_id in self.video_ids:
            try:
                video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'
                
                # Check if video folder exists
                video_folder = os.path.join(self.root_dir, f'frames/{video_label}/{video_id}/')
                
                # Check if text file exists
                text_file = os.path.join(self.root_dir, f'texts/{video_label}/{video_id}.txt')
                
                if not os.path.exists(video_folder):
                    self.logger.warning(f"Skipping {video_id}: folder not found: {video_folder}")
                    continue
                    
                frame_files = [f for f in os.listdir(video_folder) 
                              if f.endswith((".jpg", ".png"))]
                if not frame_files:
                    self.logger.warning(f"Skipping {video_id}: no frame files in {video_folder}")
                    continue
                    
                if not os.path.exists(text_file):
                    self.logger.warning(f"Skipping {video_id}: text file not found: {text_file}")
                    continue
                    
                # All checks passed, this is a valid video ID
                valid_ids.append(video_id)
                
            except Exception as e:
                print(f"Error validating {video_id}: {str(e)}")
                continue
                
        return valid_ids

    def __len__(self):
        return len(self.valid_video_ids)
        
    def __getitem__(self, index):
        video_id = self.valid_video_ids[index]
        
        embeddings = pickle.load(open(os.path.join(self.embedding_dir, f"{video_id}.pkl"), "rb"))
        image_embeddings = embeddings["image_embedding"]
        text_embeddings = embeddings["text_embedding"]
        label = embeddings["label"]

        # If shuffle_text is True, randomly select a text embedding from the other label
        if self.shuffle_text and 'non_hate' not in video_id:
            random_video_id = np.random.choice([video_id for video_id in self.valid_video_ids if 'non_hate' in video_id])
            random_embeddings = pickle.load(open(os.path.join(self.embedding_dir, f"{random_video_id}.pkl", "rb")))
            text_embeddings = random_embeddings["text_embedding"]
        elif not self.shuffle_text and 'non_hate' not in video_id:
            random_video_id = np.random.choice([video_id for video_id in self.valid_video_ids if 'hate' in video_id])
            random_embeddings = pickle.load(open(os.path.join(self.embedding_dir, f"{random_video_id}.pkl"), "rb"))
            image_embeddings = random_embeddings["image_embedding"]

        return image_embeddings, text_embeddings, label