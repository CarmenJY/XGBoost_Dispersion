#!/bin/bash
# collect_nonbonded_energies.sh
# Collects Nonbonded Force energies (El + C12 + C6) from OpenMM

echo "# Nonbonded Force Energies (El + C12 + C6 from OpenMM)"
echo "# Format: REPLICA FRAME E_nonbonded"
echo "# Generated: $(date)"
echo "#"

TEMPLATE="/expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm-template.sh"
WORK_SCRIPT="/expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm-temp-nb.sh"
BASE_DIR="/expanse/lustre/projects/uic414/bhanson2/12-6-4-NBFIX/redo-snapshots"

if [ ! -f "$TEMPLATE" ]; then
    cp /expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm.sh $TEMPLATE
fi

for replica in {1..5}; do
    echo "# Processing REPLICA-${replica}... ($(date))" >&2
    
    for frame in {1..100}; do
        cp $TEMPLATE $WORK_SCRIPT
        sed -i "s/REPLICA-XXXXX/REPLICA-${replica}/g" $WORK_SCRIPT
        sed -i "s/YYYYY/${frame}/g" $WORK_SCRIPT
        
        PDB_FILE="${BASE_DIR}/REPLICA-${replica}/MM/${frame}.pdb"
        if [ ! -f "$PDB_FILE" ]; then
            echo "${replica} ${frame} NaN"
            continue
        fi
        
        E_nonbonded=$(bash $WORK_SCRIPT 2>/dev/null | grep "NonbondedForce" | awk '{print $2}')
        
        if [ ! -z "$E_nonbonded" ]; then
            echo "${replica} ${frame} ${E_nonbonded}"
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
rm -f $WORK_SCRIPT