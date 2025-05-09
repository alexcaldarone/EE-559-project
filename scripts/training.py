import argparse
import os
from pathlib import Path
import yaml
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from src.model.models import BinaryClassifier, MultiModalClassifier
from src.model.dataset import VideoFrameTextDataset, video_batcher


# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Training pipeline")

    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Model name defined in config"
    )

    return parser.parse_args()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for frames_list, texts_list, labels in dataloader:
            frames = frames_list[0].to(device)
            text = texts_list[0].to(device)
            labels = labels.to(device).long()
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)

            logits = model(frames, text)
            labels = labels.view(-1)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def train_and_evaluate(config, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = config['train_hyperparams']
    model_config = config['models'][model_name]

    log_dir = Path("logs") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    data_root = Path(CLEAN_DATA)
    hateful_videos = [f for f in os.listdir(data_root / "frames/hateful")]
    non_hateful_videos = [f for f in os.listdir(data_root / "frames/non-hateful")]
    video_ids = hateful_videos + non_hateful_videos
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * 0.8) 
    train_ids, test_ids = video_ids[:split_idx], video_ids[split_idx:]

    # initialize clip model
    clip_model = CLIPModel.from_pretrained(config['clip']['model_name']).to(device)
    clip_processor = CLIPProcessor.from_pretrained(config['clip']['model_name'])
    clip_tokenizer = CLIPTokenizer.from_pretrained(config['clip']['model_name'])
    clip_model.eval()

    train_dataset = VideoFrameTextDataset(train_ids, preprocess=clip_processor,
                                          tokenizer=clip_tokenizer, clip_model=clip_model)
    test_dataset = VideoFrameTextDataset(test_ids, preprocess=clip_processor,
                                         tokenizer=clip_tokenizer, clip_model=clip_model)

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'],
                              shuffle=True, collate_fn=video_batcher)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, collate_fn=video_batcher)

    model = MultiModalClassifier(
        embed_dim=model_config['embed_dim'],
        num_classes=model_config['num_classes'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        transformer_hidden_dim=model_config['hidden_dim']
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=float(train_config['lr']))
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(train_config['epochs']):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for frames_list, texts_list, labels in train_loader:
            frames = frames_list[0].to(device)
            text = texts_list[0].to(device)
            labels = labels.to(device).long()
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)

            logits = model(frames, text)
            labels = labels.view(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = PROJECT_ROOT / "models/checkpoints" / f"{model_name}_{timestamp}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")

    print("Final evaluation on test set:")
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss = {final_loss:.4f}, Test Accuracy = {final_accuracy:.4f}")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    train_and_evaluate(config, "MultiModalClassifier")