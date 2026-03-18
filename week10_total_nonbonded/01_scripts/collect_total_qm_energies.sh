#!/bin/bash
# collect_total_qm_energies.sh
# Collects Total QM interaction energies from SAPT calculations

echo "# Total QM Interaction Energies (from SAPT)"
echo "# Format: REPLICA FRAME E_total_QM"
echo "# Generated: $(date)"
echo "#"

BASE_DIR="/expanse/lustre/projects/uic414/bhanson2/12-6-4-NBFIX/redo-snapshots"

for replica in {1..5}; do
    echo "# Processing REPLICA-${replica}... ($(date))" >&2
    
    for frame in {1..100}; do
        RESULTS_FILE="${BASE_DIR}/REPLICA-${replica}/SOB/snapshot-${frame}/results.txt"
        
        if [ -f "$RESULTS_FILE" ]; then
            E_total=$(grep "Total interaction energy" "$RESULTS_FILE" | tail -n 1 | awk '{print $4}')
            
            if [ ! -z "$E_total" ]; then
                echo "${replica} ${frame} ${E_total}"
            else
                echo "${replica} ${frame} NaN"
            fi
        else
            echo "${replica} ${frame} NaN"
        fi
        
        total=$((($replica - 1) * 100 + $frame))
        if [ $(($total % 50)) -eq 0 ]; then
            echo "# Progress: ${total}/500 conformers complete" >&2
        fi
    done
done

echo "# Collection complete: $(date)" >&2