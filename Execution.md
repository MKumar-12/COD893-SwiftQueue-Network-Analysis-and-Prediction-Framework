# Execution Guide for SwiftQueue ECN/RTT Project

This guide provides step-by-step instructions to execute each phase of the project — from simulation to training.

---

## 0. Environment Setup

### Create Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

Ensure Python 3.8+ and PyTorch with CUDA support is available.

---

## 1. Run Network Simulations (NS-3)
### Compile & Run
Make sure NS-3 is installed and main.cc is placed under scratch/:
```bash
cd <ns3-root>
./ns3 build
```

### Run Batch Simulations
```bash
chmod +x run_simulations.sh
./run_simulations.sh
```

- Simulations will run in parallel using **GNU Parallel**  
- Outputs go to: `Simulations_res/`  
- Logs are saved in: `Simulations_logs/`

--- 

## 2. Parse FlowMonitor XML Files
```bash
python parse_flow_xml.py
```

- Parses all `flowmonitor-results.xml` files from `../Simulations_res/`  
- Output: `grouped_flow_stats.csv` inside each simulation folder  
- Automatically deletes each XML after parsing

---

## 3. Preprocess Packet-Level CSVs
```bash
python preprocess_csv.py
```

- Input: `packet_trace.csv` from each simulation folder  
- Output: cleaned CSVs saved in `../Preprocessed/`  
- Steps include ECN encoding, RTT filtering, FlowID generation, and timestamp normalization

---

## 4. Cache CSVs into PyTorch `.pt` Format
```bash
python cache_pt.py
```

- Converts cleaned CSVs from `../Preprocessed/` to tensor format  
- Saves `.pt` files to `../CachedDatasets/`  
- Removes original CSVs after processing to save space

---

## 5. Train Transformer Model
```bash
python train.py
```
- Loads `.pt` files from `../Tensors_sml/`  
- Trains the SwiftQueueTransformer model to predict:
  - RTT (regression head)
  - ECN (binary classification head)
- Saves best model to: `swiftqueue_transformer.pth`  
- Logs metrics to: `../Logs/training_log.csv`