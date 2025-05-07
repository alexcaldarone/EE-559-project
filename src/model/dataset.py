from transformers import CLIPProcessor, CLIPModel # check if this works actually
import os
from PIL import Image
from torch.utils.data import Dataset

# Load the CLIP model and processor (evaluate whether to keep this here)
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

class VideoFrameTextDataset(Dataset):
    def __init__(self, dataframe, root_dir, preprocess, tokenizer):
        """
        dataframe: pandas DataFrame with columns ['video_id', 'text', 'label']
        root_dir: directory where video frame folders are stored (e.g., root/video_001/frame_001.jpg)
        preprocess: CLIP image transform
        tokenizer: CLIP tokenizer
        """
        self.df = dataframe
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_id, text, label = self.df.iloc[idx][['video_id', 'text', 'label']] # we need to see if this holds in the HateMMAnnotation
        video_folder = os.path.join(self.root_dir, video_id) # change this to actual video directory
        frame_files = sorted([
            os.path.join(video_folder, fname) # change this aswell because its a different folder
            for fname in os.listdir(video_folder)
            if fname.endswith((".jpg", ".png"))
        ])

        # Load and transform all frames
        frames = [self.preprocess(Image.open(f).convert("RGB")) for f in frame_files] # ok, but check if it works
        frames_tensor = torch.stack(frames)  # shape: [num_frames, 3, H, W]

        # Tokenize once, shared across all frames
        text_token = self.tokenizer([text], truncate=True)[0]  # shape: [token_length]
        
        return frames_tensor, text_token, label
    
# to determine variable batch size based on video length
def video_batcher(batch):
    # Each element is (frames_tensor, text_token, label)
    frames, texts, labels = zip(*batch)
    return frames, texts, torch.tensor(labels)