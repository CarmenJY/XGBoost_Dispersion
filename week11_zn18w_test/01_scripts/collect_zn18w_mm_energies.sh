#!/bin/bash
# collect_zn18w_mm_energies.sh
# Collects MM energies for 50 Zn-18W snapshots
# Run on Expanse: bash collect_zn18w_mm_energies.sh > zn18w_mm_energies.txt 2> zn18w_collection.log

echo "# Zn-18W MM Energies (Nonbonded + C4)"
echo "# Format: SNAPSHOT E_Nonbonded E_C4 E_Total"
echo "# Generated: $(date)"
echo "#"

TEMPLATE="/expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm-18w-template.sh"
WORK_SCRIPT="/expanse/lustre/projects/uic414/jji2/inductionXGB/run-openmm-18w-temp.sh"
BASE_DIR="/expanse/lustre/projects/uic414/bhanson2/GPR-12-6-4/snapshots-4-Madelyn/Zinc/TWO-SHELLS"

# Create template if needed
if [ ! -f "$TEMPLATE" ]; then
    cat > $TEMPLATE << 'EOF'
source ~/.bashrc

c4_prmtop=/expanse/lustre/projects/uic414/bhanson2/GPR-12-6-4/snapshots-4-Madelyn/Zinc/TWO-SHELLS/Zn-18W-2-shells-C4.prmtop
prmtop=/expanse/lustre/projects/uic414/bhanson2/GPR-12-6-4/fine-grid-Zn/MM_Portion/step-0.00/Zn-6W-TIP3P.prmtop
inpcrd=/expanse/lustre/projects/uic414/bhanson2/GPR-12-6-4/fine-grid-Zn/MM_Portion/step-0.00/Zn-6W-TIP3P.inpcrd
pdb=PDB_PLACEHOLDER

sp-12-6-4-zn.py -p $c4_prmtop -s $pdb

echo "customnonbonded is c4 only, cross check with sander"
EOF
fi

for snapshot in {1..50}; do
    echo "# Processing snapshot ${snapshot}... ($(date))" >&2
    
    PDB_FILE="${BASE_DIR}/SOB-snapshot-${snapshot}/1.pdb"
    
    if [ ! -f "$PDB_FILE" ]; then
        echo "# WARNING: PDB not found for snapshot ${snapshot}" >&2
        echo "${snapshot} NaN NaN NaN"
        continue
    fi
    
    # Create working script
    cp $TEMPLATE $WORK_SCRIPT
    sed -i "s|PDB_PLACEHOLDER|${PDB_FILE}|g" $WORK_SCRIPT
    
    # Run OpenMM and extract energies
    output=$(bash $WORK_SCRIPT 2>/dev/null)
    
    E_nb=$(echo "$output" | grep "^NonbondedForce" | awk '{print $2}')
    E_c4=$(echo "$output" | grep "^CustomNonbondedForce" | awk '{print $2}')
    
    if [ ! -z "$E_nb" ] && [ ! -z "$E_c4" ]; then
        E_total=$(echo "$E_nb + $E_c4" | bc -l)
        echo "${snapshot} ${E_nb} ${E_c4} ${E_total}"
    else
        echo "# WARNING: Could not extract energies for snapshot ${snapshot}" >&2
        echo "${snapshot} NaN NaN NaN"
    fi
    
    # Progress indicator
    if [ $(($snapshot % 10)) -eq 0 ]; then
        echo "# Progress: ${snapshot}/50 snapshots complete ($(date))" >&2
    fi
done

echo "# Collection complete: $(date)" >&2
rm -f $WORK_SCRIPT