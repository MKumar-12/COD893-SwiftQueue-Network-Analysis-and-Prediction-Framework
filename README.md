# SwiftQueue Traffic Simulation and Prediction Pipeline

This repository contains the codebase for a project focused on simulating, analyzing, and predicting network behavior in a SwiftQueue-enabled dumbbell topology using NS-3 and deep learning. It was developed as part of the COD892 course at IIT Delhi, under the supervision of Prof. Vireshwar Kumar and Prof. Tarun Mangla.

The project presents a complete pipeline for analyzing congestion control behavior across different traffic classes — including DCTCP (L4S), TCP Cubic, and cross-traffic flows — competing over a shared bottleneck link.

The system captures low-level per-packet statistics (e.g., RTT, ECN, flow metadata), processes and groups the data, and trains a custom Transformer-based model to predict:
- Round Trip Time (RTT) as a regression task
- Explicit Congestion Notification (ECN) marking as a binary classification task

This framework helps in understanding network queue dynamics and forecasting congestion responses using deep learning.

---

## Directory Structure
```
code_Submission/
├── main.cc                             # NS-3 simulation source (Dumbbell topology with ECN-enabled flows)
├── run_simulations.sh                  # Bash script to run multiple NS-3 simulations using GNU Parallel
├── parse_flow_xml.py                   # Parses flowmonitor-results.xml into grouped flow-level CSV
├── preprocess_csv.py                   # Cleans packet_trace.csv files and normalizes fields
├── cache_pt.py                         # Converts cleaned CSVs into PyTorch tensors (.pt format)
├── train.py                            # Trains the Transformer model on cached tensors
├── requirements.txt                    # Python dependencies (PyTorch, pandas, etc.)
├── swiftqueue_transformer.pth          # Saved model weights (best checkpoint)
├── dataset.py                          # Dataset classes for CSV and tensor loading
├── model/
│ └── swiftqueue_trans.py               # SwiftQueue Transformer model and optimizer
```

---

## Project Overview

### Goal
- Simulate congestion scenarios using ECN-capable flows (DCTCP, Cubic, etc.)
- Collect per-packet RTT, ECN, flow metadata using NS-3
- Train a deep learning Transformer model to predict:
  - RTT (regression)
  - ECN behavior (binary classification)

---

### Pipeline Stages
1. **Simulate traffic** in NS-3
    Simulations are executed with varying BDP, delay, and flow count using `main.cc`, which builds the SwiftQueue dumbbell topology. The `run_simulations.sh` script automates multiple simulation runs in parallel using GNU Parallel and saves the logs for analysis.

2. **Log packet and flow stats**

    Each NS-3 simulation produces:  
   - `packet_trace.csv` — logs per-packet metadata like timestamps, RTT, ECN, and flow ID  
   - `flowmonitor-results.xml` — contains aggregated flow statistics such as total delay, transmitted/received/lost packets

3. **Parse and group flows** using `parse_flow_xml.py`

    This script reads each `flowmonitor-results.xml`, extracts bidirectional flow stats, classifies TCP variants based on ports, and writes a grouped CSV (`grouped_flow_stats.csv`). It helps map flow IDs to their 5-tuple (src/dst IP/port and protocol).

4. **Preprocess CSVs** (normalize, clip, clean)

    This step cleans and normalizes the packet trace data. It converts timestamps to seconds, encodes ECN values numerically, generates Flow IDs, clips packet sizes, and filters out low-RTT packets. Output is a clean CSV per simulation for training.

5. **Convert CSVs to tensors**

    The clean CSVs are converted into sequences of fixed length (sliding window) for deep learning. Each sequence contains timestamp and packet size values, with RTT and ECN labels. These are cached as .pt files for fast training access.

6. **Train model** using a Transformer architecture

    The `SwiftQueueTransformer` is trained using the cached tensors. It predicts RTT (regression) and ECN (binary) using two heads. The training loop includes validation, early stopping, logging, and saving the best model (`swiftqueue_transformer.pth`).

---


## Author

**Manish Kumar**  
M.Tech, Department of Computer Science  
Indian Institute of Technology Delhi  
manish.kumar.iitd.cse@gmail.com

---

## Requirements

See [Execution.md](Execution.md) for complete setup and execution instructions.

## License

This project is licensed under the [MIT License](LICENSE).