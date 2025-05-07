import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import clip

from src.model.models import BinaryClassifier, MultiModalClassifier
from src.model.dataset import VideoFrameTextDataset, video_batcher


# Configs - move to a config file later
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 5
EMBED_DIM = 512
NUM_CLASSES = 2

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()  # freeze CLIP weights

# Dummy DataFrame: replace with your actual data
df = pd.DataFrame({
    'video_id': ['video_001', 'video_002'],  # subfolders inside `data/videos/`
    'text': ["some caption 1", "some caption 2"],
    'label': [0, 1]
})

# Dataset & DataLoader
dataset = VideoFrameTextDataset(df, root_dir="data/videos", preprocess=preprocess, tokenizer=clip.tokenize)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=video_batcher)

# Model
model = MultiModalClassifier(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0, 0

    for frames_list, texts_list, labels in dataloader:
        # B = 1, so unpack
        frames = frames_list[0].to(DEVICE)       # [num_frames, 3, H, W]
        text = texts_list[0].unsqueeze(0).to(DEVICE)  # [1, seq_len]
        labels = labels.to(DEVICE)

        # No gradients for CLIP encoders
        with torch.no_grad():
            image_embeds = clip_model.encode_image(frames)  # [F, D]
            text_embed = clip_model.encode_text(text)       # [1, D]

        # Repeat text embedding for all frames
        text_embeds = text_embed.repeat(image_embeds.size(0), 1)  # [F, D]

        # Add batch dim for model
        image_embeds = image_embeds.unsqueeze(0)  # [1, F, D]
        text_embeds = text_embeds.unsqueeze(0)    # [1, F, D]

        logits = model(image_embeds, text_embeds)  # [1, num_classes]

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct

    acc = total_correct / len(dataloader)
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")
