import random
import yaml

import numpy as np
import torch
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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, drop_intermediate=False)

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