import argparse
import os
from pathlib import Path
import yaml
import random
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from src.model.models import BinaryClassifier, MultiModalClassifier, CrossModalFusion
from src.model.dataset import PrecomputedEmbeddingsDataset, video_batcher
from src.utils.logger import setup_logger


logger = setup_logger(f"training_{datetime.now().strftime('%Y%m%d_%H%M')}")

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"
EMBEDDINGS_DIR = PROJECT_ROOT / "data/embeddings"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_FUNCTION_DICT = {
    "gelu": torch.nn.GELU
}

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
            #logits = logits.mean(dim = 0).unsqueeze(0)
            #labels = labels.view(-1)
            labels = torch.tile(labels, (logits.size(0), 1)).squeeze(1).long()
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

    log_dir = Path("logs") / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    data_root = Path(CLEAN_DATA)
    hateful_videos = [f for f in os.listdir(data_root / "frames/hateful")]
    non_hateful_videos = [f for f in os.listdir(data_root / "frames/non-hateful")]
    video_ids = hateful_videos + non_hateful_videos
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * 0.8) 
    train_ids, test_ids = video_ids[:split_idx], video_ids[split_idx:]

    train_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=EMBEDDINGS_DIR,
        split=train_ids
    )
    test_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=EMBEDDINGS_DIR,
        split=test_ids
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'],
        shuffle=True, 
        collate_fn=video_batcher
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        collate_fn=video_batcher)
    
    if model_name == "MultiModalClassifier":
        model = MultiModalClassifier(
            embed_dim=model_config['embed_dim'],
            num_classes=model_config['num_classes'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            transformer_hidden_dim=model_config['hidden_dim'],
            classifier_hidden_dims=model_config["classifier_hidden_dims"],
            dropout_rate=model_config["dropout_rate"],
            activation_function=LOSS_FUNCTION_DICT[model_config["activation_function"]]
        ).to(device)
    elif model_name == "CrossModalFusion":
        model = CrossModalFusion(
            embed_dim=model_config["embed_dim"],
            num_heads=model_config["num_heads"],
            num_classes=2    
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config['lr']),
        weight_decay=float(train_config.get('weight_decay', 0.0))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['epochs'],
        eta_min=0.0
    )
    criterion = torch.nn.CrossEntropyLoss()

    history = []

    for epoch in range(train_config['epochs']):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for frames_list, texts_list, labels in train_loader:
            frames = frames_list[0].to(device) # why index zero here?
            text = texts_list[0].to(device)
            labels = labels.to(device).long()
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            
            logits = model(frames, text)
            #logits = logits.mean(dim = 0).unsqueeze(0)
            labels = torch.tile(labels, (logits.size(0), 1)).squeeze(1).long()
            #print(f"logits shape: {logits.shape}, labels shape: {labels.shape}")
            #print("logits", logits, logits.shape)
            #print("labels", labels, labels.shape)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)
        
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        logger.info(f"Epoch {epoch+1}/{train_config['epochs']} "
                    f"Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f} | "
                    f"Test Loss={test_loss:.4f}, Test Accurary={test_accuracy:.4f}")
        
        #print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
        #      f"Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': test_loss,
            'val_acc': test_accuracy,
            'lr': scheduler.get_last_lr()[0]
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = PROJECT_ROOT / "models/checkpoints" / f"{model_name}_{timestamp}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Saved checkpoint to {weights_path}")
    
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)
    logger.info(f"Final evaluation on test set | Test Loss = {final_loss:.4f}, Test Accuracy = {final_accuracy:.4f}")

    df = pd.DataFrame(history)
    results_path = PROJECT_ROOT / "results" / f"{model_name}_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"Saved training history to {results_path}")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    train_and_evaluate(config, args.model_name)