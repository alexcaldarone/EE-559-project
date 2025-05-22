import argparse
import os
from pathlib import Path
import random
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


from src.model.models import (
    BinaryClassifier, 
    MultiModalClassifier, 
    CrossModalFusion,
    BinaryClassifierNoImage,
    BinaryClassifierNoText
)
from src.model.dataset import (
    PrecomputedEmbeddingsDataset, 
    RandomAdversarialVideoTextDataset,
    AugmentedDataset,
    video_batcher
)
from src.utils.logger import setup_logger
from src.utils.training_utils import (
    set_seed,
    load_config,
    evaluate
)

logger = setup_logger(f"training_{datetime.now().strftime('%Y%m%d_%H%M')}")

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data/raw"
CLEAN_DATA = PROJECT_ROOT / "data/clean"
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "models/checkpoints"
AUGMENTED_DATA_DIR = PROJECT_ROOT / "data/embeddings_transformed"

LOSS_FUNCTION_DICT = {"gelu": torch.nn.GELU}

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
    parser.add_argument(
        "--test_on_shuffled",
        action="store_true",
        help="Determine whether we test on the adversarially swapped data."
    )
    parser.add_argument(
        "--modality_to_shuffle_in_test",
        type=str,
        required=False,
        choices=["text", "image"],
        help="Modality to swap when testing with the andversarial shuffling"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Whether this tranining run uses finefuned embeddings or not"
    )
    parser.add_argument(
        "--use-augmented-data",
        action="store_true",
        help="Determine whether to use augmented data for training"
    )
    args = parser.parse_args()

    if args.test_on_shuffled and not args.modality_to_shuffle_in_test:
        parser.error(
            "--modality_to_shuffle_in_test is required when --test_on_shuffled is set"
        )
    
    return args

def main():
    args = parse_args()
    config = load_config(args.config)
    seeds = config['train_hyperparams'].get('seeds', [42])

    if args.finetune:
        EMBEDDINGS_DIR = PROJECT_ROOT / "data/finetuned_embeddings"
    else:
        EMBEDDINGS_DIR = PROJECT_ROOT / "data/embeddings"
    
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
        elif args.model_name == "BinaryClassifierNoImage":
            model = BinaryClassifierNoImage(model_config['embedding_size'])
        elif args.model_name == "BinaryClassifierNoText":
            model = BinaryClassifierNoText(model_config['embedding_size'])
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

        if args.use_augmented_data:
            train_ds = AugmentedDataset(
                embedding_dir1=AUGMENTED_DATA_DIR,
                embedding_dir2=EMBEDDINGS_DIR,
                prob=0.8, # change this to be param from config file
                split=train_ids
            )
        else:
            train_ds = PrecomputedEmbeddingsDataset(EMBEDDINGS_DIR, train_ids)
        
        if args.test_on_shuffled:
            shuffle_text = True if args.modality_to_shuffle_in_test == "text" else False
            test_ds = RandomAdversarialVideoTextDataset(test_ids, logger, device, EMBEDDINGS_DIR, shuffle_text=shuffle_text)
        else:
            test_ds = PrecomputedEmbeddingsDataset(EMBEDDINGS_DIR, test_ids)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=train_config['batch_size'], shuffle=True, collate_fn=video_batcher
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1, shuffle=False, collate_fn=video_batcher
        )

        # define paths to save model weights and result csv
        subdir = 'train_normal_test_on_shuffled' if args.test_on_shuffled else 'training_simple'
        finetune_suffix = '_finetuned' if args.finetune else ''
        shuffle_suffix  = '_shuffled'  if args.test_on_shuffled else ''
        if args.test_on_shuffled:
            modality_shuffle_suffix = '_text' if args.modality_to_shuffle_in_test == "text" else '_image'
        else:
            modality_shuffle_suffix = ''
        model_run_dir = MODEL_CHECKPOINTS_DIR / subdir / f"{args.model_name}_{timestamp}{finetune_suffix}{shuffle_suffix}"
        model_run_dir.mkdir(parents=True, exist_ok=True)

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
        ckpt_name = f"seed{seed}_epoch{epoch+1}.pt"
        ckpt_path = model_run_dir / ckpt_name
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved model weights to {ckpt_path}")

    # Save results
    results_dir = PROJECT_ROOT / 'results' / subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    # saving result csv
    fname = f"{args.model_name}_{timestamp}{finetune_suffix}{shuffle_suffix}{modality_shuffle_suffix}.csv"
    out_path = results_dir / fname
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()