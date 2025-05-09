from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA = PROJECT_ROOT / "data/clean"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoFrameTextDataset(Dataset):
    def __init__(
            self, 
            video_ids, 
            preprocess, 
            tokenizer, 
            clip_model,
            logger,
            device,
            root_dir=CLEAN_DATA
        ):
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
        self.clip_model = clip_model

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

    def __getitem__(self, idx):
        video_id = self.valid_video_ids[idx]
        video_label = 'non-hateful' if 'non_hate' in video_id else 'hateful'

        video_folder = self.root_dir + f'/frames/{video_label}/{video_id}/' 
        try:
            frame_files = sorted([
                os.path.join(video_folder, fname)
                for fname in os.listdir(video_folder)
                if fname.endswith((".jpg", ".png"))
            ])

            if not frame_files:
                raise FileNotFoundError(f"No frame files found for {video_id}")
        except Exception as e:
            return self.__geitem__((idx + 1) % len(self.valid_video_ids))


        # Load Text
        text_file = self.root_dir + f'/texts/{video_label}/{video_id}.txt'
        try:
            with open(text_file, 'r') as f:
                text = f.read().strip()

            #print("text before preprocessing", text)
            if not text:
                self.logger.warning(f"Warning: Empty text for {video_id}, using placeholder.")
                text = self.empty_text_placeholder
        except Exception as e:
            #print(f"Error loading text for {video_id}: {str(e)}")
            text = self.empty_text_placeholder
        
        video_label = 1 if video_label == 'hateful' else 0

        # Chunk the text into blocks of 50 words and tokenize
        text = text.split()
        # is there a way i can dinamically chunk text here?
        text = [' '.join(text[i:i + min(30,len(text))]) for i in range(0, len(text), 30)]
        # dont neeed this 
        #text = [self.tokenizer(text_chunk, return_tensors="pt", padding=True).to(DEVICE) for text_chunk in text]
        
        #print(len(text))
        #print("text", text)

        # Load and transform all frames
        try:
            # Load and transform all frames
            frames = []
            for f in frame_files:
                try:
                    img = Image.open(f).convert("RGB")
                    frames.append(img)
                except Exception as e:
                    self.logger.warning(f"Error loading image {f}: {str(e)}")
                    # Skip bad frames
                    continue
            
            # Make sure we have at least one frame
            if not frames:
                raise ValueError("No valid frames loaded")
            
            inputs = []
            for frame in frames:
                try:
                    processed = self.preprocess(
                        text=text,
                        images=frame,
                        return_tensors="pt",
                        padding=True).to(self.device)
                    inputs.append(processed)
                except Exception as e:
                    self.logger.warning(f"Error preprocessing: {str(e)}")
                    continue
            
            # Make sure we have at least one valid input
            if not inputs:
                raise ValueError("No valid inputs created after preprocessing")
                
            with torch.no_grad():
                outputs = []
                for input_item in inputs:
                    try:
                        output = self.clip_model(**input_item)
                        outputs.append(output)
                    except Exception as e:
                        self.logger.warning(f"Error in model forward pass: {str(e)}")
                        continue
            
            if not outputs:
                raise ValueError("No valid outputs from model")
                
            # Extract embeddings
            image_embeddings = torch.stack([output.image_embeds for output in outputs])
            
            # Safe mean operation - handle potential empty tensors
            text_embedding = torch.stack([
                output.text_embeds.mean(dim=0, keepdim=True) 
                if output.text_embeds.size(0) > 0 
                else torch.zeros_like(output.text_embeds[0]).unsqueeze(0) 
                for output in outputs
            ])
            
            video_label_tensor = torch.full((len(outputs), 1), video_label, dtype=torch.float)
            
            return image_embeddings, text_embedding, video_label_tensor
            
        except Exception as e:
            print(f"Fatal error processing {video_id}: {str(e)}")
            # Return a different valid video as fallback
            return self.__getitem__((idx + 1) % len(self.valid_video_ids))


# to determine variable batch size based on video length
def video_batcher(batch):
    # Each element is (frames_tensor, text_token, label)
    # Filter out any None values in case an item failed
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        # Create an empty batch with appropriate dimensions
        # You'll need to define appropriate dimensions based on your model
        empty_frames = torch.zeros((0, 512))  # Adjust dimensions as needed
        empty_text = torch.zeros((0, 512))  # Adjust dimensions as needed
        empty_labels = torch.zeros((0, 1))
        return empty_frames, empty_text, empty_labels
    
    frames, texts, labels = zip(*valid_batch)
    return frames, texts, torch.cat(labels, dim=0)