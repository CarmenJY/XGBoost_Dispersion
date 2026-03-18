"""
Extract geometric features from Zn-18W PDB files
Uses the same feature_extraction.py from Week 2
Run from DISPML directory: python week11_zn18w_test/01_scripts/extract_zn18w_features.py
"""

import sys
import json
from pathlib import Path

# Import your existing feature extraction
sys.path.append(str(Path.cwd()))
from feature_extraction import extract_features, atoms_to_mol_dict, read_xyz

print("="*70)
print("EXTRACTING FEATURES FROM ZN-18W SNAPSHOTS")
print("="*70)

# Input/output paths
pdb_dir = Path("week11_zn18w_test/02_raw_data/zn18w_pdbs")
output_dir = Path("week11_zn18w_test/02_raw_data/features")
output_dir.mkdir(parents=True, exist_ok=True)

# Process all 50 snapshots
features_list = []
snapshot_ids = []

for i in range(1, 51):
    pdb_file = pdb_dir / f"snapshot_{i}.pdb"
    
    if not pdb_file.exists():
        print(f"⚠ Warning: Missing {pdb_file}")
        continue
    
    print(f"Processing snapshot {i}...", end=" ")
    
    try:
        # Your extract_features takes an XYZ path directly
        # First convert PDB to XYZ format
        xyz_file = output_dir / f"snapshot_{i}.xyz"
        
        # Read PDB and write as XYZ
        with open(pdb_file) as f_in:
            atoms = []
            coords = []
            for line in f_in:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_type = line[76:78].strip()
                    if not atom_type:
                        atom_type = line[12:16].strip()[0]
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atoms.append(atom_type)
                    coords.append([x, y, z])
        
        # Write XYZ file
        with open(xyz_file, 'w') as f_out:
            f_out.write(f"{len(atoms)}\n")
            f_out.write(f"Zn-18W snapshot {i}\n")
            for atom, (x, y, z) in zip(atoms, coords):
                f_out.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")
        
        # Extract features using your function
        features = extract_features(str(xyz_file), compute_baseline=False)
        
        # Save individual JSON
        output_file = output_dir / f"snapshot_{i}.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        features_list.append(features)
        snapshot_ids.append(i)
        
        print("✓")
        
    except Exception as e:
        print(f"✗ Error: {e}")

print(f"\n✓ Extracted features for {len(features_list)} snapshots")
print(f"✓ Saved to: {output_dir}/")

# Save summary
summary = {
    'num_snapshots': len(snapshot_ids),
    'snapshot_ids': snapshot_ids,
    'features': features_list
}

summary_file = Path("week11_zn18w_test/02_raw_data/features_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved to: {summary_file}")
print("\n" + "="*70)
print("Feature extraction complete!")
print("="*70)