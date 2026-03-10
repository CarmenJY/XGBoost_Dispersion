"""
Load QM and MM energies, compute residuals, prepare for ML training.
Run from DISPML directory: python week3_ml_training/01_scripts/load_and_analyze.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
QM_FILE = Path("week8_ml_training/02_raw_data/qm_energies.txt")
MM_FILE = Path("week8_ml_training/02_raw_data/mm_energies.txt")
OUTPUT_DIR = Path("week8_ml_training/03_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 1: Load QM Energies (E_ref)")
print("="*60)

# Load QM energies
qm_data = []
with open(QM_FILE) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        replica, frame, e_qm = int(parts[0]), int(parts[1]), float(parts[2])
        qm_data.append((replica, frame, e_qm))

E_QM = np.array([x[2] for x in qm_data])
print(f"✓ Loaded {len(E_QM)} QM energies")
print(f"  Range: {E_QM.min():.2f} to {E_QM.max():.2f}")
print(f"  Mean: {E_QM.mean():.2f}")
print(f"  Std: {E_QM.std():.2f}")

print("\n" + "="*60)
print("STEP 2: Load MM Energies (E_baseline from OpenMM)")
print("="*60)

# Load MM energies (TODO: update this once MM data is collected)
if MM_FILE.exists():
    mm_data = []
    with open(MM_FILE) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            replica, frame, e_mm = int(parts[0]), int(parts[1]), float(parts[2])
            mm_data.append((replica, frame, e_mm))
    
    E_MM = np.array([x[2] for x in mm_data])
    print(f"✓ Loaded {len(E_MM)} MM energies")
    print(f"  Range: {E_MM.min():.2f} to {E_MM.max():.2f}")
    print(f"  Mean: {E_MM.mean():.2f}")
    print(f"  Std: {E_MM.std():.2f}")
else:
    print("⚠ MM energies not yet collected")
    print("  Using baseline_c4 from Week 2 as temporary substitute...")
    
    # Load from Week 2 JSON files
    E_MM = []
    for replica in range(1, 6):
        for frame in range(1, 101):
            json_file = Path(f"output/REPLICA-{replica}_{frame}.json")
            with open(json_file) as f:
                data = json.load(f)
                E_MM.append(data['E_baseline_C4'])
    
    E_MM = np.array(E_MM)
    print(f"✓ Loaded {len(E_MM)} baseline energies from Week 2")
    print(f"  Range: {E_MM.min():.2f} to {E_MM.max():.2f}")
    print(f"  Mean: {E_MM.mean():.2f}")

print("\n" + "="*60)
print("STEP 3: Compute Residuals")
print("="*60)

# Compute residuals
Delta_E = E_QM - E_MM
baseline_pct = abs(E_MM.mean() / E_QM.mean()) * 100

print(f"Residual (ΔE = E_QM - E_MM):")
print(f"  Min: {Delta_E.min():.2f}")
print(f"  Max: {Delta_E.max():.2f}")
print(f"  Mean: {Delta_E.mean():.2f}")
print(f"  Std: {Delta_E.std():.2f}")
print(f"\nBaseline Performance:")
print(f"  MM captures {baseline_pct:.1f}% of QM induction energy")

print("\n" + "="*60)
print("STEP 4: Visualize")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Energy distributions
axes[0,0].hist(E_QM, bins=30, alpha=0.7, label='E_QM (SAPT)', color='blue')
axes[0,0].hist(E_MM, bins=30, alpha=0.7, label='E_MM (OpenMM)', color='orange')
axes[0,0].set_xlabel('Energy (kcal/mol)')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()
axes[0,0].set_title('Energy Distributions')
axes[0,0].grid(alpha=0.3)

# 2. Residual distribution
axes[0,1].hist(Delta_E, bins=30, color='green', alpha=0.7)
axes[0,1].axvline(Delta_E.mean(), color='red', linestyle='--', 
                label=f'Mean: {Delta_E.mean():.2f}')
axes[0,1].set_xlabel('Residual ΔE (kcal/mol)')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()
axes[0,1].set_title('Residual Distribution')
axes[0,1].grid(alpha=0.3)

# 3. MM vs QM scatter
axes[1,0].scatter(E_MM, E_QM, alpha=0.5, s=20)
min_val = min(E_MM.min(), E_QM.min())
max_val = max(E_MM.max(), E_QM.max())
axes[1,0].plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='y=x')
axes[1,0].set_xlabel('E_MM (kcal/mol)')
axes[1,0].set_ylabel('E_QM (kcal/mol)')
axes[1,0].legend()
axes[1,0].set_title('MM vs QM Energies')
axes[1,0].grid(alpha=0.3)

# 4. Residual vs MM
axes[1,1].scatter(E_MM, Delta_E, alpha=0.5, s=20, color='purple')
axes[1,1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1,1].set_xlabel('E_MM (kcal/mol)')
axes[1,1].set_ylabel('Residual ΔE (kcal/mol)')
axes[1,1].set_title('Residual vs MM Energy')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "energy_analysis.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_plot}")

print("\n" + "="*60)
print("STEP 5: Save Data for ML")
print("="*60)

# Save for ML training
output_data = OUTPUT_DIR / "ml_training_data.npz"
np.savez(output_data,
        E_QM=E_QM,
        E_MM=E_MM,
        Delta_E=Delta_E,
        replica=[x[0] for x in qm_data],
        frame=[x[1] for x in qm_data])

print(f"✓ Saved: {output_data}")
print("\n" + "="*60)
print("COMPLETE! Ready for ML training.")
print("="*60)