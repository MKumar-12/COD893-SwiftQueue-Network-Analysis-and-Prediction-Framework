# Filename: run_simulations.sh
# Author: Manish Kumar
#
# Description:
# -----------
# This script runs a series of simulations using NS-3, varying parameters such as BDP, delay, and flow multipliers.
# It generates a log for each simulation and a master log summarizing all simulations.
# It uses GNU Parallel to run multiple simulations concurrently.
#
# Usage:
# ------
# 1. Ensure NS-3 is installed and the path to the NS-3 executable is set correctly.
# 2. Make the script executable: chmod +x run_simulations.sh
# 3. Run the script: ./run_simulations.sh
#
# Note:
# -----
# NS3_EXEC: Path to the NS-3 executable (default is "./ns3").
# SCRIPT_NAME: Path to the main simulation script (default is "scratch/main.cc").
# LOG_DIR: Directory where simulation logs will be saved (default is "Simulations_logs").
#
# Parameters:
# ---------
# - simNum: Simulation number for output directory.
# - BDP_MULTIPLIERS: Array of multipliers for the Bandwidth-Delay Product (BDP).
# - DELAYS: Array of propagation delays in milliseconds.
# - FLOW_MULTIPLIERS: Array of multipliers for the number of flows.
#
# Contact:
# --------
# manish.kumar.iitd.cse@gmail.com

#!/bin/bash

# Configurable Parameters
BDP_MULTIPLIERS=(2 5 10 15 20)
DELAYS=(20 40 100 150 200)
FLOW_MULTIPLIERS=(0.5 1 2.5 5 25)

# Paths
NS3_EXEC="./ns3"
SCRIPT_NAME="scratch/main.cc"

# Output folder for logs
LOG_DIR="Simulations_logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/simulation_master.log"
> "$MASTER_LOG"  # Clear it at the beginning
export MASTER_LOG

# Prepare input combinations
combinations_file="sim_combinations.txt"
> "$combinations_file"
simNum=1
for bdp in "${BDP_MULTIPLIERS[@]}"; do
    for delay in "${DELAYS[@]}"; do
        for flows in "${FLOW_MULTIPLIERS[@]}"; do
            echo "$simNum $bdp $delay $flows" >> "$combinations_file"
            ((simNum++))
        done
    done
done

# Export NS3 executable and script path
export NS3_EXEC SCRIPT_NAME LOG_DIR

# Function to run a single simulation
run_sim() {
    simNum=$1
    bdp=$2
    delay=$3
    flows=$4

    timestamp=$(date)
    timestamp_safe=$(date "+%Y%m%d_%H-%M-%S")
    logFile="${LOG_DIR}/sim${simNum}_bdp${bdp}_delay${delay}_flows${flows}_${timestamp_safe}.log"

    echo "[${timestamp}] Running simNum=${simNum} | BDP=${bdp} | Delay=${delay} | Flows=${flows}" | tee -a "$MASTER_LOG"
    $NS3_EXEC run "${SCRIPT_NAME} --bdpMultiplier=${bdp} --delayMs=${delay} --numFlowsMultiplier=${flows} --simNum=${simNum}" > "$logFile" 2>&1

    echo "[`date`] Finished simNum=${simNum}" | tee -a "$MASTER_LOG"
}
export -f run_sim

# Run simulations in parallel
cat "$combinations_file" | parallel -j 6 --colsep ' ' run_sim {1} {2} {3} {4}

echo "All simulations complete. Logs saved in $LOG_DIR/"