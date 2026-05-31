"""
AuthNet Training Pipeline
Trains the EmbeddingNet using Triplet Margin Loss with semi-hard negative mining.
Uses pytorch-metric-learning for miners, losses, and samplers.
"""

import os
import sys
import json
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import EmbeddingNet, build_model
from src.dataset import create_dataloaders
from src.evaluate import compute_recall_at_k

from pytorch_metric_learning import losses, miners


def train_one_epoch(
    model: EmbeddingNet,
    dataloader: torch.utils.data.DataLoader,
    loss_func,
    miner,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Train for one epoch.
    
    Returns:
        dict with 'loss', 'num_triplets', 'time' for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_triplets = 0
    num_batches = 0
    
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", 
                leave=False, ncols=100)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        embeddings = model(images)
        
        # Mine hard/semi-hard triplets from the batch
        hard_pairs = miner(embeddings, labels)
        
        # Compute triplet loss on mined pairs
        loss = loss_func(embeddings, labels, hard_pairs)
        
        # Skip if no valid triplets found
        if loss.item() == 0 and len(hard_pairs[0]) == 0:
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_triplets = len(hard_pairs[0]) if len(hard_pairs) > 0 else 0
        total_triplets += num_triplets
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'triplets': num_triplets,
        })
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'loss': avg_loss,
        'num_triplets': total_triplets,
        'num_batches': num_batches,
        'time': elapsed,
    }


@torch.no_grad()
def validate(
    model: EmbeddingNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Validate model by computing Recall@K on the test/validation set.
    
    Returns:
        dict with recall values and embedding extraction info.
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Validating", leave=False, ncols=100):
        images = images.to(device)
        embeddings = model(images)
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute Recall@K
    recall_dict = compute_recall_at_k(
        embeddings=all_embeddings.numpy(),
        labels=all_labels.numpy(),
        k_values=config.RECALL_K,
    )
    
    return recall_dict


def train(
    resume_from: Optional[str] = None,
    num_epochs: Optional[int] = None,
):
    """
    Full training pipeline.
    
    Args:
        resume_from: Path to checkpoint to resume from.
        num_epochs: Override number of epochs.
    """
    num_epochs = num_epochs or config.NUM_EPOCHS
    device = config.DEVICE
    
    print("=" * 60)
    print("  AuthNet Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Margin: {config.MARGIN}")
    print(f"  Mining: {config.MINING_TYPE}")
    print("=" * 60)
    
    # ── Data ──
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders()
    
    if len(train_dataset) == 0:
        print("\n[ERROR] No training data found! Run download_data.py first.")
        print(f"  Expected data at: {config.COMBINED_DIR}")
        return
    
    # ── Model ──
    if resume_from and os.path.exists(resume_from):
        from src.model import load_model
        model = load_model(resume_from, device)
        model.train()
        print(f"Resumed from {resume_from}")
    else:
        model = build_model(device)
    
    # ── Loss & Miner ──
    loss_func = losses.TripletMarginLoss(margin=config.MARGIN)
    miner = miners.TripletMarginMiner(
        margin=config.MARGIN,
        type_of_triplets=config.MINING_TYPE,
    )
    
    # ── Optimizer ──
    param_groups = model.get_parameter_groups()
    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    # ── Scheduler ──
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # ── Training Loop ──
    best_recall_at_1 = 0.0
    patience_counter = 0
    training_log = []
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_func=loss_func,
            miner=miner,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        
        # Validate
        val_metrics = validate(model, test_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        # Current LRs
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # Log
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_triplets': train_metrics['num_triplets'],
            'train_time': train_metrics['time'],
            **{f'recall@{k}': v for k, v in val_metrics.items()},
            'lr_backbone': current_lrs[0],
            'lr_head': current_lrs[1] if len(current_lrs) > 1 else current_lrs[0],
        }
        training_log.append(epoch_log)
        
        recall_at_1 = val_metrics.get(1, 0.0)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Triplets: {train_metrics['num_triplets']:5d} | "
              f"R@1: {recall_at_1:.2%} | "
              f"R@4: {val_metrics.get(4, 0.0):.2%} | "
              f"Time: {train_metrics['time']:.1f}s")
        
        # ── Save best model ──
        if recall_at_1 > best_recall_at_1:
            best_recall_at_1 = recall_at_1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall_at_1': best_recall_at_1,
                'config': {
                    'embedding_dim': config.EMBEDDING_DIM,
                    'backbone': config.BACKBONE,
                    'margin': config.MARGIN,
                },
            }
            torch.save(checkpoint, config.BEST_MODEL_PATH)
            print(f"  * New best R@1: {best_recall_at_1:.2%} -- saved to {config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
        
        # ── Early stopping ──
        if patience_counter >= config.PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {config.PATIENCE} epochs. "
                  f"Best R@1: {best_recall_at_1:.2%}")
            break
    
    # ── Save last model ──
    last_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_recall_at_1': best_recall_at_1,
    }
    torch.save(last_checkpoint, config.LAST_MODEL_PATH)
    
    # ── Save training log ──
    log_path = os.path.join(config.OUTPUT_DIR, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best Recall@1:  {best_recall_at_1:.2%}")
    print(f"  Best model:     {config.BEST_MODEL_PATH}")
    print(f"  Last model:     {config.LAST_MODEL_PATH}")
    print(f"  Training log:   {log_path}")
    print("=" * 60)
    
    return model, training_log


if __name__ == "__main__":
    train()
