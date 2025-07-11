# SPDX-License-Identifier: MIT
# © 2025 Manish Kumar

"""    
    Filename: preprocess_csv.py

    Description:
    -----------
    Preprocesses packet trace CSV files generated by NS-3 simulations,
    extracting essential fields, encoding ECN markings, and generating a FlowID.
    
    - Reads packet trace CSV files from a specified input directory.
    - Normalizes timestamps and prepares flow identifiers.
    - Encodes ECN markings into numerical values.
    - Clips packet sizes to a standard range.
    - Normalizes RTT values and filters out packets with RTT < 3 x propagation delay.
    - Runs in parallel for multiple CSV files.
    - Saves the preprocessed data to a specified output directory.
    
    Usage:
    -------
    # usage: python preprocess_csv.py                                               -> Preprocess all packet_trace.csv files in parallel
    
    Note:
    -------
    - INPUT_DIR: Directory containing subdirectories with packet_trace.csv files.
    - OUTPUT_DIR: Directory where preprocessed CSV files will be saved.
    - The script expects the input directory to have a specific structure with subdirectories named after simulations, containing packet_trace.csv file.

    - Run the script to preprocess all packet_trace.csv files in the input directory.
    - The output will be saved in the specified OUTPUT_DIR directory, with each simulation's data in a separate CSV file.
    
    Contact:
    -------
    manish.kumar.iitd.cse@gmail.com
"""

import os
import glob
import re
import pandas as pd
from joblib import Parallel, delayed


INPUT_DIR = "../Simulations_res/"
OUTPUT_DIR = "../Preprocessed/"


# Encode ECN_Marking
ECN_MAP = {
    'NotECT': 0,
    'L4S': 1,
    'CE': 2
}



# Function to extract propagation delay from filename
def extract_propagation_delay(filename):
    match = re.search(r'delay(\d+)', filename)
    if match:
        return int(match.group(1)) / 1000.0  # Convert ms → seconds
    else:
        raise ValueError(f"Could not extract delay from filename: {filename}")

# Function to preprocess packet trace CSV files
def preprocess_packet_trace(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['SrcIP', 'SrcPort', 'DstIP', 'DstPort', 'Protocol'])                 # Ensure no NaN in key columns

    # Normalize timestamp relative to first packet in the file (in seconds)
    df['Timestamp'] = (df['Timestamp_ns'] - df['Timestamp_ns'].min()) / 1e9                     # convert ns → seconds

    # Prepare flow column
    df['Flow'] = df.apply(
        lambda row: f"({row['SrcIP']},{row['SrcPort']},{row['DstIP']},{row['DstPort']},{row['Protocol']})",
        axis=1
    )

    # Generate FlowID based on 5-tuple
    df['FlowID'] = df.groupby(
        ['SrcIP', 'SrcPort', 'DstIP', 'DstPort', 'Protocol']
    ).ngroup()

    # Filter out non-TX direction   
    # df = df[df['Direction'] == 'TX']

    # Hot-encoding ECN_Marking
    df['ECN_Marking'] = df['ECN_Marking'].map(ECN_MAP).fillna(-1).astype(int)

    # Clip PacketSize to standard range
    df['PacketSize'] = df['PacketSize'].clip(64, 1500).astype('float32')

    # Normalize RTT to seconds
    df['RTT'] = df['RTT_us'] / 1e6

    # Extract propagation delay from directory name (in seconds)
    prop_delay_sec = extract_propagation_delay(os.path.basename(os.path.dirname(input_path)))

    # Drop rows with RTT < 3 × propagation delay
    df = df[df['RTT'] >= 3 * prop_delay_sec]
    
    # Keeping only essential fields
    df = df[['Timestamp', 'PacketSize', 'ECN_Marking', 'Flow', 'FlowID', 'RTT']]

    # Save the preprocessed DataFrame to CSV
    df.to_csv(output_path, index=False)
    print(f"Preprocessed and saved: {output_path}")

# Function to run preprocessing in parallel
def run_parallel_preprocessing():
    # Prepare args for parallel
    tasks = []
    
    for input_file in input_files:
        # Extract simulation dir name
        sim_dir_name = os.path.basename(os.path.dirname(input_file))
        output_file = os.path.join(OUTPUT_DIR, f"{sim_dir_name}.csv")
        tasks.append((input_file, output_file))

    # Parallel execution
    Parallel(n_jobs=-1)(
        delayed(preprocess_packet_trace)(in_path, out_path)
        for in_path, out_path in tasks
    )

    print(f"All CSV files preprocessed and saved to {OUTPUT_DIR}")



# Driver code
if __name__ == "__main__":
    # Ensure output directory exists, create if not
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # Ensure input directory exists
    if not os.path.exists(INPUT_DIR):   
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
        
    # Find all packet_trace.csv files in the input directory
    input_files = glob.glob(os.path.join(INPUT_DIR, "*/packet_trace.csv"))
    print(f"Found {len(input_files)} CSV files to preprocess.")

    run_parallel_preprocessing()