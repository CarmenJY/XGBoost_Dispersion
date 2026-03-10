#!/bin/bash
# collect_mm_energies.sh
# Collects MM induction energies (CustomNonbondedForce) from OpenMM runs
# Run this ON Expanse - will take 30-60 minutes for 500 conformers!
#
# Usage: bash collect_mm_energies.sh > mm_energies.txt 2> mm_collection.log

echo "# MM Induction Energies (CustomNonbondedForce from OpenMM)"
echo "# Format: REPLICA FRAME E_MM"
echo "# Generated: $(date)"
echo "#"

TEMPLATE="/expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm-template.sh"
WORK_SCRIPT="/expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm-temp.sh"
BASE_DIR="/expanse/lustre/projects/uic414/bhanson2/12-6-4-NBFIX/redo-snapshots"

# Create template from original (only once)
if [ ! -f "$TEMPLATE" ]; then
    cp /expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm.sh $TEMPLATE
fi

for replica in {1..5}; do
    echo "# Processing REPLICA-${replica}... ($(date))" >&2
    
    for frame in {1..100}; do
        # Create working copy and substitute values
        cp $TEMPLATE $WORK_SCRIPT
        sed -i "s/REPLICA-XXXXX/REPLICA-${replica}/g" $WORK_SCRIPT
        sed -i "s/YYYYY/${frame}/g" $WORK_SCRIPT
        
        # Check if PDB exists
        PDB_FILE="${BASE_DIR}/REPLICA-${replica}/MM/${frame}.pdb"
        if [ ! -f "$PDB_FILE" ]; then
            echo "# WARNING: PDB not found for REPLICA-${replica} frame ${frame}" >&2
            echo "${replica} ${frame} NaN"
            continue
        fi
        
        # Run OpenMM and extract CustomNonbondedForce value
        E_MM=$(bash $WORK_SCRIPT 2>/dev/null | grep "CustomNonbondedForce" | awk '{print $2}')
        
        if [ ! -z "$E_MM" ]; then
            echo "${replica} ${frame} ${E_MM}"
        else
            echo "# WARNING: Could not extract E_MM from REPLICA-${replica} frame ${frame}" >&2
            echo "${replica} ${frame} NaN"
        fi
        
        # Progress indicator every 50 frames
        total=$((($replica - 1) * 100 + $frame))
        if [ $(($total % 50)) -eq 0 ]; then
            echo "# Progress: ${total}/500 conformers complete ($(date))" >&2
        fi
    done
done

echo "# Collection complete: $(date)" >&2

# Cleanup
rm -f $WORK_SCRIPT