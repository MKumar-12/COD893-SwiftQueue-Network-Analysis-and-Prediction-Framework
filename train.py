"""
    Filename: train.py
    Author: Manish Kumar

    Description:
    -----------
    This script trains the SwiftQueueTransformer model on preprocessed packet trace data.
    It loads cached tensors from disk, performs training and validation, and logs the results.
    It uses a sliding window approach to create sequences of fixed length from the packet data.
    The model predicts both RTT values and ECN labels for each sequence.
    
    The training process includes:
        - Loading cached tensors from .pt files 
        - Creating training and validation datasets
        - Training the SwiftQueueTransformer model
        - Evaluating the model on validation data
        - Saving the best model based on F1 score

    We track:
        Accuracy of predicted ECN labels
        F1-accuracy to assess performance
    
    Usage:
    -------
    # usage: python train.py

    Note:
    -----
    - CACHED_ROOT: Directory containing cached .pt files.
    - TRAINING_LOG_DIR: Directory where training logs will be saved.
    - SEQ_LEN: Length of the sliding window for sequences.
    - PATIENCE: Number of epochs to wait for improvement before early stopping.
    - The script expects the cached tensors to have specific keys: 'sequences', 'rtt_targets', and 'ecn_labels'.
    - The model will be saved as 'swiftqueue_transformer.pth' in the current directory.
    - The training log will be saved as 'training_log.csv' in the TRAINING_LOG_DIR directory.

    Contact:
    -------
    manish.kumar.iitd.cse@gmail.com
"""

import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import CachedTensorDataset
from model.swiftqueue_trans import SwiftQueueTransformer, SwiftQueueOptimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from tqdm import tqdm

# Params
BATCH_SIZE = 8
EPOCHS = 10
SEQ_LEN = 50
PATIENCE = 3                                    # for early stopping
CACHED_ROOT = '../Tensors/'                     # Path to cached .pt files
TRAINING_LOG_DIR = '../Logs/'                   # Directory to save training logs


# Driver function
if __name__ == "__main__":
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all .pt cached files
    pt_files = [os.path.join(CACHED_ROOT, f) for f in os.listdir(CACHED_ROOT) if f.endswith('.pt')]
    print(f"Found {len(pt_files)} cached .pt files in {CACHED_ROOT}")

    # Create datasets from all .pt files
    print("Loading cached datasets...")
    datasets = []
    for pt in tqdm(pt_files, desc="Loading PT files"):
        try:
            datasets.append(CachedTensorDataset(pt))
        except Exception as e:  
            print(f"[WARNING] Skipping {pt} due to load error: {e}")
    dataset = ConcatDataset(datasets)
    total_sequences = sum(len(ds) for ds in datasets)
    print(f"Total sequences across all CSVs: {total_sequences}")

    # Train-validation split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training sequences: {train_size} | Validation sequences: {val_size}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)                    # Input size: [BATCH_SIZE, SEQ_LEN, 2]
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Training batches per epoch: {len(train_loader)} | Validation batches: {len(val_loader)}")


    # Initialize model and optimizer
    model = SwiftQueueTransformer(feature_size=2).to(device)                       # PacketSize, Timestamp
    optimizer = SwiftQueueOptimizer(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {type(model).__name__} with {num_params:,} trainable parameters")

    # Print model architecture
    print("-" * 50)
    print("Model architecture:")
    print(model)
    print("-" * 50)

    # Training loop with early stopping and logging
    best_f1 = 0.0
    no_improve_epochs = 0
    train_log = []
    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        model.train()
        total_loss, total_rtt_loss, total_ecn_loss = 0.0, 0.0, 0.0
        num_batches = 0

        for batch in train_loader:
            inputs, (rtt_targets, ecn_labels) = batch
            inputs = inputs.to(device)
            rtt_targets = rtt_targets.to(device)
            ecn_labels = ecn_labels.to(device)

            loss_dict = optimizer.partial_fit((inputs, (rtt_targets, ecn_labels)))
            total_loss += loss_dict["loss"]
            total_rtt_loss += loss_dict["rtt_loss"]
            total_ecn_loss += loss_dict["ecn_loss"]
            num_batches += 1

        # Evaluate on validation set
        model.eval()
        all_ecn_preds, all_ecn_labels = [], []
        all_rtt_preds, all_rtt_true = [], [] 
        with torch.no_grad():
            for batch in val_loader:
                inputs, (rtt_targets, ecn_labels) = batch
                inputs = inputs.to(device)
                rtt_targets = rtt_targets.to(device)
                ecn_labels = ecn_labels.to(device)

                rtt_preds, ecn_logits = model(inputs)
                
                # ECN evaluation
                preds = (torch.sigmoid(ecn_logits) > 0.5).int().cpu().numpy()
                true = ecn_labels.int().cpu().numpy()
                all_ecn_preds.extend(preds.flatten())
                all_ecn_labels.extend(true.flatten())

                # RTT evaluation
                rtt_preds = rtt_preds.squeeze(-1).cpu().numpy()
                rtt_true = rtt_targets.cpu().numpy()
                all_rtt_preds.extend(rtt_preds.flatten())
                all_rtt_true.extend(rtt_true.flatten())

        # Aggregate results
        acc = accuracy_score(all_ecn_labels, all_ecn_preds)
        f1 = f1_score(all_ecn_labels, all_ecn_preds)
        rtt_mse = mean_squared_error(all_rtt_true, all_rtt_preds)
        rtt_mae = mean_absolute_error(all_rtt_true, all_rtt_preds)

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1:02d} completed in {epoch_time:.2f} seconds")

        # Logging
        print(f"Epoch {epoch+1:02d} | Avg Loss: {total_loss/num_batches:.4f} | RTT: {total_rtt_loss/num_batches:.4f} | ECN: {total_ecn_loss/num_batches:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | RTT_MAE: {rtt_mae:.4f} | RTT_MSE: {rtt_mse:.4f}")
        train_log.append({
            'epoch': epoch + 1,
            'loss': total_loss / num_batches,
            'rtt_loss': total_rtt_loss / num_batches,
            'ecn_loss': total_ecn_loss / num_batches,
            'acc': acc,
            'f1': f1,
            'RTT_MAE': rtt_mae,
            'RTT_MSE': rtt_mse,
        })

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'swiftqueue_transformer.pth')
            print("Model improved and saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs. Early stopping.")
                break


    # Save training log
    if not os.path.exists(TRAINING_LOG_DIR):
        os.makedirs(TRAINING_LOG_DIR)

    train_log_path = os.path.join(TRAINING_LOG_DIR, 'training_log.csv')
    pd.DataFrame(train_log).to_csv(train_log_path, index=False)
    print("Training log saved to training_log.csv.")