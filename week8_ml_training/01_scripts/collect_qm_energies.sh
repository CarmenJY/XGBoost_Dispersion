#!/bin/bash
# collect_qm_energies.sh
# 
# Collects QM induction energies (E_orb) from all 500 conformers on Expanse
# Run this script ON Expanse
#
# Usage: bash collect_qm_energies.sh > qm_energies.txt

echo "# QM Induction Energies (E_orb) from SAPT"
echo "# Format: REPLICA FRAME E_QM_induction"
echo "# Generated: $(date)"
echo "#"

BASE_DIR="/expanse/lustre/projects/uic414/bhanson2/12-6-4-NBFIX/redo-snapshots"

# Loop through all 5 replicas
for replica in {1..5}; do
    echo "# Processing REPLICA-${replica}..." >&2
    
    # Loop through all 100 frames
    for frame in {1..100}; do
        RESULTS_FILE="${BASE_DIR}/REPLICA-${replica}/SOB/snapshot-${frame}/results.txt"
        
        # Check if file exists
        if [ -f "$RESULTS_FILE" ]; then
            # Extract the last "Orbital (E_orb):" value
            E_QM=$(grep "Orbital (E_orb):" "$RESULTS_FILE" | tail -n 1 | awk '{print $3}')
            
            if [ ! -z "$E_QM" ]; then
                echo "${replica} ${frame} ${E_QM}"
            else
                echo "# WARNING: Could not extract E_QM from REPLICA-${replica} frame ${frame}" >&2
                echo "${replica} ${frame} NaN"
            fi
        else
            echo "# WARNING: Missing file for REPLICA-${replica} frame ${frame}" >&2
            echo "${replica} ${frame} NaN"
        fi
    done
done

echo "# Collection complete: $(date)" >&2