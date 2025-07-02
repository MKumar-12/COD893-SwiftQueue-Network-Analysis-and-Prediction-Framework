# SPDX-License-Identifier: MIT
# © 2025 Manish Kumar

"""
    Filename: dataset.py
    
    Description:
    -----------
    This module defines a PyTorch Dataset for processing packet trace data.
    It reads a CSV file containing packet traces, extracts relevant features,
    groups packets by flow, and prepares sequences for training a model.
    
    The dataset supports sliding window sequences of fixed length and provides
    both features and targets (RTT and ECN labels) for each sequence.
    
    It also includes a cached dataset class for loading preprocessed tensors
    from disk, which can be used to speed up training by avoiding repeated
    parsing of CSV files.

    Usage:
    -------
    - Initialize `PacketDataset` with a CSV file and sequence length.
    - Use `DataLoader` to create batches for training.
    - Use `CachedTensorDataset` to load preprocessed tensors from disk.

    Note:
    -----
    Ensure that the CSV file contains the required columns: 'FlowID', 'PacketSize',
    'ECN_Marking', 'RTT', and 'Timestamp'. The dataset will raise an error if
    any of these columns are missing.   

    Contact:
    -------
    manish.kumar.iitd.cse@gmail.com
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

class PacketDataset(Dataset):
    def __init__(self, csv_file, sequence_length=50):
        df = pd.read_csv(csv_file, on_bad_lines="skip")
        df = df.dropna()

        # Ensure required fields are present
        required = {'FlowID', 'PacketSize', 'ECN_Marking', 'RTT', 'Timestamp'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {csv_file}. Required: {required}")

        self.sequences = []
        self.rtt_targets = []
        self.ecn_labels = []

        # Group by Flow ID and pad/clip to sequence_length
        for _, flow_df in df.groupby('FlowID'):
            # Ensure the flow has enough packets
            if len(flow_df) < sequence_length:
                continue

            # Sliding window approach to create sequences
            for i in range(len(flow_df) - sequence_length):
                seq = flow_df.iloc[i:i+sequence_length]

                # Extract features and labels for each window
                features = seq[['PacketSize', 'Timestamp']].values.astype('float32')                    # Extracting PacketSize and Timestamp
                rtt = seq['RTT'].values.astype('float32')
                ecn = seq['ECN_Marking'].values.astype('float32')

                self.sequences.append(torch.tensor(features))                # tensor [50, 2]
                self.rtt_targets.append(torch.tensor(rtt))                   # tensor [50]       
                self.ecn_labels.append(torch.tensor(ecn))                    # tensor [50]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], (self.rtt_targets[idx], self.ecn_labels[idx])
    

class CachedTensorDataset(Dataset):
    def __init__(self, pt_path):
        print(f"Loading: {pt_path}")
        try:
            data = torch.load(pt_path)
        except Exception as e:
            print(f"[ERROR] Failed to load {pt_path}: {e}")
            raise   

        self.sequences = data['sequences']
        self.rtt_targets = data['rtt_targets']
        self.ecn_labels = data['ecn_labels']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], (self.rtt_targets[idx], self.ecn_labels[idx])