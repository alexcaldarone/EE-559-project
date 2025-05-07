from transformers import CLIPProcessor, CLIPModel # check if this works actually
import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA = PROJECT_ROOT / "data/clean"

# Load the CLIP model and processor (evaluate whether to keep this here)
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

class VideoFrameTextDataset(Dataset):
    def __init__(self, video_id, preprocess, tokenizer, root_dir=CLEAN_DATA):
        """
        video_id: unique identifier for each video e.g. 'hate_video_1_snippet_0'
        root_dir: directory where video frame folders are stored (by default ../data/clean)
        preprocess: CLIP image transform
        tokenizer: CLIP tokenizer
        """
        self.video_id = video_id
        self.root_dir = str(root_dir)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_label = 'non-hateful' if 'non_hate' in self.video_id else 'hateful'

        video_folder = os.path.join(self.root_dir, f'/frames/{video_label}/{self.video_id}') 
        frame_files = sorted([
            os.path.join(video_folder, fname)
            for fname in os.listdir(video_folder)
            if fname.endswith((".jpg", ".png"))
        ])

        with open(os.path.join(self.root_dir, f'/texts/{video_label}/{self.video_id}.txt'), 'r') as f:
            text = f.read().strip()

        # Load and transform all frames
        frames = [self.preprocess(Image.open(f).convert("RGB")) for f in frame_files] # ok, but check if it works
        frames_tensor = torch.stack(frames)  # shape: [num_frames, 3, H, W]

        # Tokenize once, shared across all frames
        text_token = self.tokenizer([text], truncate=True)[0]  # shape: [token_length]
        
        return frames_tensor, text_token, video_label
    
# to determine variable batch size based on video length
def video_batcher(batch):
    # Each element is (frames_tensor, text_token, label)
    frames, texts, labels = zip(*batch)
    return frames, texts, torch.tensor(labels)