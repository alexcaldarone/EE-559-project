from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Prepare a simple dataset class
class VideoDataset(Dataset):
    def __init__(self, image_folder, text_file, label, clip_processor):
        self.image_folder = image_folder
        self.text_file = text_file
        self.label = label
        self.clip_processor = clip_processor
        self.images = self.load_images(image_folder)
        self.text = self.load_text(text_file)
    
    def load_images(self, folder):
        # Load all images from the folder
        image_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        image_paths = [os.path.join(folder, file) for file in image_files]
        images = [Image.open(path) for path in image_paths]
        return images
    
    def load_text(self, text_file):
        # Load text (transcript) from the text file
        with open(text_file, 'r') as file:
            text = file.read().strip()
        return text
    
    def __len__(self):
        # Return the number of images (frames)
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get the image and corresponding text for this index
        image = self.images[idx]
        text = self.text
        
        # Process the image and text to get embeddings
        inputs = self.clip_processor(text=text, images=image, return_tensors="pt", padding=True)
        
        # Get the embeddings from the CLIP model
        image_embedding = clip_model.get_image_features(**inputs)
        text_embedding = clip_model.get_text_features(**inputs)
        
        # Flatten the embeddings into a vector
        return image_embedding.squeeze(0), text_embedding.squeeze(0), torch.tensor(self.label, dtype=torch.float)

# Function to prepare data from folder and text file
def prepare_data(image_folder, text_file, label):
    dataset = VideoDataset(image_folder, text_file, label, clip_processor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

# Assuming you have a list of images and corresponding text transcripts
# images = [Image.open(image_path) for image_path in image_paths]  # load your images
# texts = ["some text"] * len(images)  # your text data (transcripts)
# labels = [1, 0, 1, 0]  # binary labels (0 or 1)

