from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA = PROJECT_ROOT / "data/clean"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model and processor (evaluate whether to keep this here)
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)

class VideoFrameTextDataset(Dataset):
    def __init__(self, video_ids, preprocess, tokenizer, root_dir=CLEAN_DATA):
        """
        video_id: unique identifier for each video e.g. 'hate_video_1_snippet_0'
        root_dir: directory where video frame folders are stored (by default ../data/clean)
        preprocess: CLIP image transform
        tokenizer: CLIP tokenizer
        """
        self.video_ids = video_ids
        self.root_dir = str(root_dir)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_label = 'non-hateful' if 'non_hate' in self.video_ids[idx] else 'hateful'

        video_folder = self.root_dir + f'/frames/{video_label}/{self.video_ids[idx]}/' 
        frame_files = sorted([
            os.path.join(video_folder, fname)
            for fname in os.listdir(video_folder)
            if fname.endswith((".jpg", ".png"))
        ])

        # Load Text
        with open(self.root_dir + f'/texts/{video_label}/{self.video_ids[idx]}.txt', 'r') as f:
            text = f.read().strip()

        video_label = 1 if video_label == 'hateful' else 0

        # Chunk the text into blocks of 50 words and tokenize
        text = text.split()
        text = [' '.join(text[i:i + min(50,len(text))]) for i in range(0, len(text), 50)]
        # dont neeed this 
        #text = [self.tokenizer(text_chunk, return_tensors="pt", padding=True).to(DEVICE) for text_chunk in text]
        
        print(len(text))

        # Load and transform all frames
        frames = [Image.open(f).convert("RGB") for f in frame_files]

        inputs = [self.preprocess(
            text= text,
            images=frame,
            return_tensors="pt",
            padding=True).to(DEVICE) for frame in frames]

        with torch.no_grad():
            # need to find better presentation for this -> right now we are treating the sliced text as independent inputs   
            outputs = [clip_model(**input) for input in inputs]

        image_embeddings = torch.stack([output.image_embeds for output in outputs])
        text_embedding = torch.stack([output.text_embeds for output in outputs]) 

        return image_embeddings, text_embedding, video_label
    
# to determine variable batch size based on video length
def video_batcher(batch):
    # Each element is (frames_tensor, text_token, label)
    frames, texts, labels = zip(*batch)
    return frames, texts, torch.tensor(labels)

if __name__ == "__main__":
    # Example usage
    dataset = VideoFrameTextDataset(video_ids=['hate_video_1_snippet_0'], preprocess=clip_processor, tokenizer=clip_tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=video_batcher)

    for frames, texts, labels in dataloader:
        print(f"Frames: {type(frames)}, Texts: {type(texts)}, Labels: {labels.dim()}")
        print(f"Frames shape: {frames[0].shape}, Texts shape: {texts[0].shape}")

    