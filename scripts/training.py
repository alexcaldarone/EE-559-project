import argparse
import os
from pathlib import Path
import yaml
import random
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)

from src.model.models import BinaryClassifier, MultiModalClassifier, CrossModalFusion
from src.model.dataset import PrecomputedEmbeddingsDataset, video_batcher
from src.utils.logger import setup_logger


logger = setup_logger(f"training_{datetime.now().strftime('%Y%m%d_%H%M')}")

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"
EMBEDDINGS_DIR = PROJECT_ROOT / "data/embeddings" # change to finetuned_embeddings for finetuning
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "models/checkpoints"

LOSS_FUNCTION_DICT = {"gelu": torch.nn.GELU}


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    total_loss = total_correct = total_samples = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for frames_list, texts_list, labels in dataloader:
            frames = frames_list[0].to(device)
            text = texts_list[0].to(device)
            labels = labels.to(device).long()
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)

            logits = model(frames, text)
            labels = torch.tile(labels, (logits.size(0), 1)).squeeze(1).long()
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            # check these work
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='binary')
    kappa = cohen_kappa_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, labels=[0,1], average=None)
    recall = recall_score(all_labels, all_preds, labels=[0,1], average=None)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float('nan')
    pr_auc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'kappa': kappa,
        'precision_0': precision[0],
        'precision_1': precision[1],
        'recall_0': recall[0],
        'recall_1': recall[1],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    seeds = config['train_hyperparams'].get('seeds', [42])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_cols = [
        'seed', 'epoch',
        'train_loss', 'train_accuracy', 'train_f1', 'train_kappa',
        'train_precision_0', 'train_precision_1', 'train_recall_0', 'train_recall_1',
        'train_roc_auc', 'train_pr_auc', 'train_specificity',
        'train_fpr', 'train_tpr', 'train_thresholds',
        'test_loss', 'test_accuracy', 'test_f1', 'test_kappa',
        'test_precision_0', 'test_precision_1', 'test_recall_0', 'test_recall_1',
        'test_roc_auc', 'test_pr_auc', 'test_specificity',
        'test_fpr', 'test_tpr', 'test_thresholds',
        'lr'
    ]
    results_df = pd.DataFrame(columns=results_cols)

    for seed in seeds:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_config = config['train_hyperparams']
        model_config = config['models'][args.model_name]

        model = None
        if args.model_name == "MultiModalClassifier":
            model = MultiModalClassifier(
                embed_dim=model_config['embed_dim'],
                num_classes=model_config['num_classes'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                transformer_hidden_dim=model_config['hidden_dim'],
                classifier_hidden_dims=model_config['classifier_hidden_dims'],
                dropout_rate=model_config['dropout_rate'],
                activation_function=LOSS_FUNCTION_DICT[model_config['activation_function']]
            )
        elif args.model_name == "CrossModalFusion":
            model = CrossModalFusion(
                embed_dim=model_config['embed_dim'],
                num_heads=model_config['num_heads'],
                num_classes=2
            )
        else: # binary classifier
            model = BinaryClassifier(model_config['embedding_size'])

        logger.info("Model architecture:\n%s", model)
        logger.info("Training hyperparameters: %s", train_config)
        logger.info("Model hyperparameters: %s", model_config)

        model = model.to(device)
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

        # Prepare data
        data_root = Path(CLEAN_DATA)
        hateful = [f for f in os.listdir(data_root / 'frames/hateful')]
        non_hateful = [f for f in os.listdir(data_root / 'frames/non-hateful')]
        video_ids = hateful + non_hateful
        random.shuffle(video_ids)
        split_idx = int(len(video_ids) * 0.8)
        train_ids, test_ids = video_ids[:split_idx], video_ids[split_idx:]

        train_loader = DataLoader(
            PrecomputedEmbeddingsDataset(EMBEDDINGS_DIR, train_ids),
            batch_size=train_config['batch_size'], shuffle=True, collate_fn=video_batcher
        )
        test_loader = DataLoader(
            PrecomputedEmbeddingsDataset(EMBEDDINGS_DIR, test_ids),
            batch_size=1, shuffle=False, collate_fn=video_batcher
        )

        # Training loop with metrics
        for epoch in range(train_config['epochs']):
            model.train()
            for frames_list, texts_list, labels in train_loader:
                frames = frames_list[0].to(device)
                text = texts_list[0].to(device)
                labels = labels.to(device).long()
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)

                logits = model(frames, text)
                labels = torch.tile(labels, (logits.size(0), 1)).squeeze(1).long()
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Evaluate on train and test
            train_metrics = evaluate(model, train_loader, criterion, device)
            test_metrics = evaluate(model, test_loader, criterion, device)
            lr = scheduler.get_last_lr()[0]

            
            logger.info(
                f"Epoch {epoch+1}/{train_config['epochs']} "
                f"Train F1={train_metrics['f1']:.4f}, Test F1={test_metrics['f1']:.4f}"
            )

            # Record
            row = {
                'seed': seed,
                'epoch': epoch,
                'lr': lr
            }
            # Merge train and test metrics
            for prefix, metrics in [('train', train_metrics), ('test', test_metrics)]:
                for k, v in metrics.items():
                    row[f"{prefix}_{k}"] = v
            results_df.loc[len(results_df)] = row
    
        # save final model
        ckpt_path = MODEL_CHECKPOINTS_DIR / f"{args.model_name}_seed{seed}_epoch{epoch+1}_{timestamp}.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved model to {ckpt_path}")

    # Save results
    out_path = PROJECT_ROOT / 'results' / f"{args.model_name}_{timestamp}.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()