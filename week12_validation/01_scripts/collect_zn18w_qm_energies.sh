#!/bin/bash
# collect_zn18w_qm_energies.sh
# Collects QM reference energies for Zn-18W from SOB-EDA results

echo "# Zn-18W QM Reference Energies (Total interaction energy from SOB-EDA)"
echo "# Format: SNAPSHOT E_QM"
echo "# Generated: $(date)"
echo "#"

BASE_DIR="/expanse/lustre/projects/uic414/bhanson2/GPR-12-6-4/snapshots-4-Madelyn/Zinc/TWO-SHELLS"

for snapshot in {1..50}; do
    RESULTS_FILE="${BASE_DIR}/SOB-snapshot-${snapshot}/results.txt"
    
    if [ -f "$RESULTS_FILE" ]; then
        E_QM=$(grep "Total interaction energy" "$RESULTS_FILE" | awk '{print $4}')
        
        if [ ! -z "$E_QM" ]; then
            echo "${snapshot} ${E_QM}"
        else
            echo "${snapshot} NaN"
        fi
    else
        echo "${snapshot} NaN"
    fi
    
    if [ $(($snapshot % 10)) -eq 0 ]; then
        echo "# Progress: ${snapshot}/50" >&2
    fi
done

echo "# Collection complete: $(date)" >&2