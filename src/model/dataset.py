from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA = PROJECT_ROOT / "data/clean"
EMBEDDINGS_DATA = PROJECT_ROOT / "data/embeddings"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrecomputedEmbeddingsDataset(Dataset):
    def __init__(
            self,
            embeddings_dir,
            split=None,
            split_ratio=0.8,
            train=True,
            embed_dim=512
        ):
        """
        Dataset class for precomputed CLIP embeddings
        
        Args:
            embeddings_dir: Directory containing precomputed embeddings
            split: Optional list of video_ids to use (for train/val/test splits)
            split_ratio: Ratio for train/val split if split is not provided
            train: Whether to use train or validation split if splitting by ratio
        """
        self.embeddings_dir = embeddings_dir
        self.embedding_dimension = embed_dim
        
        # Load the index file
        with open(os.path.join(embeddings_dir, "index.pkl"), 'rb') as f:
            self.index = pickle.load(f)
            
        # Apply split if provided
        if split is not None:
            self.index = [item for item in self.index if item['video_id'] in split]
        # Otherwise split by ratio
        elif split_ratio < 1.0:
            total = len(self.index)
            split_idx = int(total * split_ratio)
            if train:
                self.index = self.index[:split_idx]
            else:
                self.index = self.index[split_idx:]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]
        
        # Load the embeddings file
        with open(item['file'], 'rb') as f:
            data = pickle.load(f)
            
        # Extract embeddings and label
        image_embeddings = data['image_embeddings']
        B = image_embeddings.size(0)
        text_embedding = torch.tile(data['text_embedding'], (B, 1, 1))
        label = torch.tensor([[data['label']]], dtype=torch.float)
        
        return image_embeddings, text_embedding, label

class RandomAdversarialVideoTextDataset(Dataset):
    def __init__(
            self, 
            video_ids,
            logger,
            device,
            embedding_dir,
            shuffle_text = True, # True if shuffle hateful texts into non-hateful texts, 
                              # False to shuffle videos instead
        ):
        """
        Dataset class used for the training epochs that contain the examples of hateful data 
        points with a swapped modality. 

        video_ids: unique identifiers for each video e.g. 'hate_video_1_snippet_0'
        embedding_dir: directory where video frame folders are stored (by default ../data/clean)
        shuffle_text: True if shuffle hateful texts into non-hateful texts, 
                      False to shuffle videos instead
        """
        self.video_ids = video_ids
        self.embeddings_dir = str(embedding_dir)
        self.root_dir = CLEAN_DATA

        self.logger = logger
        self.device = device

        self.shuffle_text = shuffle_text
        self.empty_text_placeholder = "empty"

        # Load the index file
        with open(os.path.join(self.embeddings_dir, "index.pkl"), 'rb') as f:
            self.index = pickle.load(f)

        self.index = [item for item in self.index if item['video_id'] in video_ids]

    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, index):
        video_id = self.index[index]
        
        embeddings = pickle.load(open(video_id['file'], "rb"))
        image_embeddings = embeddings["image_embeddings"]
        text_embeddings = embeddings["text_embedding"]
        label = embeddings["label"]

        # If shuffle_text is True, randomly select a text embedding from the other label
        if self.shuffle_text and 'non_hate' not in video_id:
            random_video_id = np.random.choice([video_id for video_id in self.index if video_id['label'] == 0])
            random_embeddings = pickle.load(open(random_video_id['file'], "rb"))
            text_embeddings = random_embeddings["text_embedding"]
        elif not self.shuffle_text and 'non_hate' not in video_id:
            random_video_id = np.random.choice([video_id for video_id in self.index if video_id['label'] == 1])
            random_embeddings = pickle.load(open(random_video_id['file'], "rb"))
            image_embeddings = random_embeddings["image_embeddings"]
        
        B = image_embeddings.size(0)
        text_embeddings = torch.tile(text_embeddings, (B, 1, 1))
        label = torch.tensor([[label]], dtype=torch.float)

        return image_embeddings, text_embeddings, label

# not used in project
class AugmentedDataset(Dataset):
    def __init__(
            self,
            embedding_dir1,
            embedding_dir2,
            prob,
            split,
            split_ratio=None,
            train=True,
            embed_dim=512
        ):
        self.dataset1 = PrecomputedEmbeddingsDataset(
            embeddings_dir=embedding_dir1,
            split=split,
            split_ratio=split_ratio,
            train=train,
            embed_dim=embed_dim
        )

        self.dataset2 = PrecomputedEmbeddingsDataset(
            embeddings_dir=embedding_dir2,
            split=split,
            split_ratio=split_ratio,
            train=train,
            embed_dim=embed_dim
        )

        self.p = prob

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, index):
        if np.random.random() < self.p:
            # sample uniformly from dataset1
            rand_idx = np.random.randint(len(self.dataset1))
            return self.dataset1[rand_idx]
        else:
            # sample uniformly from dataset2
            rand_idx = np.random.randint(len(self.dataset2))
            return self.dataset2[rand_idx]

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
        # (downstream if this can't be done for a particular transcript the datapoint is 
        # excluded from the training set)
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

# to determine variable batch size based on video length (for precomputed embeddings datasets)
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

# to determine variable batch size based on video length (for raw videos datasets)
def raw_video_collate_fn(batch, processor):
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
