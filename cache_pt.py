# SPDX-License-Identifier: MIT
# © 2025 Manish Kumar

"""
    Filename: cache_pt.py

    Description:
    ----------- 
    This script processes all CSV files in a specified directory, converts them into PyTorch tensors,
    and saves them as .pt files in a designated output directory. 
    It uses parallel processing to speed up the conversion process.

    - Each CSV file is processed to extract sequences, RTT targets, and ECN labels.
    - The sequences are padded or clipped to a fixed length (SEQ_LEN).
    - The processed tensors are saved in the OUTPUT_DIR directory with the same base name as the CSV file.
    - The original CSV files are deleted after processing to save space.

    Usage:
    -------
    # usage: python cache_pt.py

    Note:
    -----
    - CSV_ROOT: Directory containing preprocessed CSV files.
    - OUTPUT_DIR: Directory where cached tensors will be saved.
    - The script expects the CSV files to have specific columns: 'FlowID', 'PacketSize', 'ECN_Marking', 'RTT', and 'Timestamp'.
    - The OUTPUT_DIR will be created if it does not exist.

    Contact:
    --------
    manish.kumar.iitd.cse@gmail.com
"""

from dataset import PacketDataset
import os
import torch
from pathlib import Path
from joblib import Parallel, delayed

SEQ_LEN = 50
CSV_ROOT = '../Preprocessed/'                    # Path to preprocessed CSV files
OUTPUT_DIR = '../CachedDatasets'                 # Directory to save cached tensors
NUM_WORKERS = 8                                  # To parallel  ize


# Function to process a single CSV
def process_csv(csv_path):
    try:
        dataset = PacketDataset(csv_path, sequence_length=SEQ_LEN)
        output_path = os.path.join(OUTPUT_DIR, Path(csv_path).stem + '.pt')
        torch.save({
            'sequences': dataset.sequences,
            'rtt_targets': dataset.rtt_targets,
            'ecn_labels': dataset.ecn_labels,
        }, output_path)
        print(f"[PID {os.getpid()}] Cached {len(dataset)} sequences from {csv_path} → {output_path}")

        # Delete the CSV file after processing
        try:
            os.remove(csv_path)
        except Exception as e:
            print(f"Error deleting CSV file {csv_path}: {e}")
    
    except Exception as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")



# Driver function
if __name__ == "__main__":
    # Load all csv files from CSV_ROOT directory
    csv_files = [os.path.join(CSV_ROOT, f) for f in os.listdir(CSV_ROOT) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {CSV_ROOT}")

    # Create output dir., if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    Parallel(n_jobs=NUM_WORKERS)(
        delayed(process_csv)(csv) for csv in csv_files
    )

    print(f"All CSV files processed and cached in {OUTPUT_DIR}")