"""
Use Week 10 trained model to predict corrections for Zn-18W snapshots
Run from DISPML directory: python week11_zn18w_test/01_scripts/predict_zn18w.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb

print("="*70)
print("WEEK 11: PREDICTING ZN-18W ENERGIES WITH WEEK 10 MODEL")
print("="*70)

# ============================================================
# STEP 1: Load Week 10 Trained Model
# ============================================================
print("\n[1/5] Loading Week 10 trained model...")

model_file = Path("week10_total_nonbonded/04_models/xgboost_model.json")
model = xgb.XGBRegressor()
model.load_model(str(model_file))

print(f"✓ Loaded model from: {model_file}")

# Load feature names
with open("week10_total_nonbonded/04_models/feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

print(f"✓ Model uses {len(feature_names)} features")

# ============================================================
# STEP 2: Load Zn-18W MM Energies
# ============================================================
print("\n[2/5] Loading Zn-18W MM energies...")

mm_data = {}
with open("week11_zn18w_test/02_raw_data/zn18w_mm_energies.txt") as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        snapshot = int(parts[0])
        e_nb = float(parts[1])
        e_c4 = float(parts[2])
        e_total = float(parts[3])
        mm_data[snapshot] = {
            'E_nonbonded': e_nb,
            'E_c4': e_c4,
            'E_MM_total': e_total
        }

print(f"✓ Loaded MM energies for {len(mm_data)} snapshots")
print(f"  Range: {min(d['E_MM_total'] for d in mm_data.values()):.2f} to {max(d['E_MM_total'] for d in mm_data.values()):.2f} kcal/mol")

# ============================================================
# STEP 3: Load Zn-18W Features
# ============================================================
print("\n[3/5] Loading Zn-18W features...")

features_dir = Path("week11_zn18w_test/02_raw_data/features")
features_matrix = []
valid_snapshots = []

for snapshot in sorted(mm_data.keys()):
    feature_file = features_dir / f"snapshot_{snapshot}.json"
    
    if not feature_file.exists():
        print(f"⚠ Warning: Missing features for snapshot {snapshot}")
        continue
    
    with open(feature_file) as f:
        feats = json.load(f)
        features_matrix.append([feats[fn] for fn in feature_names])
        valid_snapshots.append(snapshot)

X = np.array(features_matrix)

print(f"✓ Loaded features for {len(X)} snapshots")
print(f"  Feature matrix shape: {X.shape}")

# ============================================================
# STEP 4: Predict Corrections
# ============================================================
print("\n[4/5] Predicting ΔE corrections...")

Delta_E_predicted = model.predict(X)

print(f"✓ Generated predictions")
print(f"  Predicted ΔE range: {Delta_E_predicted.min():.2f} to {Delta_E_predicted.max():.2f} kcal/mol")
print(f"  Predicted ΔE mean: {Delta_E_predicted.mean():.2f} ± {Delta_E_predicted.std():.2f} kcal/mol")

# ============================================================
# STEP 5: Calculate Final Predictions
# ============================================================
print("\n[5/5] Calculating final energy predictions...")

results = []
for i, snapshot in enumerate(valid_snapshots):
    mm_energy = mm_data[snapshot]['E_MM_total']
    delta_e = float(Delta_E_predicted[i])  # Convert to Python float
    final_energy = mm_energy + delta_e
    
    results.append({
        'snapshot': int(snapshot),  # Ensure int
        'E_MM': float(mm_energy),   # Ensure float
        'Delta_E_ML': delta_e,
        'E_predicted': float(final_energy)
    })

# Save results
output_dir = Path("week11_zn18w_test/04_predictions")
output_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / "predictions.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save as table
table_file = output_dir / "predictions_table.txt"
with open(table_file, 'w') as f:
    f.write("# Zn-18W Energy Predictions Using Week 10 Model\n")
    f.write("# Snapshot | E_MM | Delta_E_ML | E_predicted\n")
    f.write("#" + "-"*60 + "\n")
    for r in results:
        f.write(f"{r['snapshot']:3d} {r['E_MM']:10.3f} {r['Delta_E_ML']:10.3f} {r['E_predicted']:10.3f}\n")

print(f"✓ Saved predictions to: {results_file}")
print(f"✓ Saved table to: {table_file}")

# ============================================================
# Create Visualizations
# ============================================================
print("\n[6/6] Creating visualizations...")

E_MM = np.array([r['E_MM'] for r in results])
Delta_E = np.array([r['Delta_E_ML'] for r in results])
E_pred = np.array([r['E_predicted'] for r in results])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Distribution of corrections
axes[0,0].hist(Delta_E, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[0,0].axvline(Delta_E.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {Delta_E.mean():.2f}')
axes[0,0].set_xlabel('Predicted ΔE (kcal/mol)', fontsize=11)
axes[0,0].set_ylabel('Count', fontsize=11)
axes[0,0].set_title('Distribution of ML Corrections', fontsize=12, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# Plot 2: MM vs Predicted
axes[0,1].scatter(E_MM, E_pred, alpha=0.6, s=50)
axes[0,1].plot([E_MM.min(), E_MM.max()], [E_MM.min(), E_MM.max()], 
               'r--', linewidth=2, label='No correction')
axes[0,1].set_xlabel('E_MM (kcal/mol)', fontsize=11)
axes[0,1].set_ylabel('E_predicted (MM + ML) (kcal/mol)', fontsize=11)
axes[0,1].set_title('MM Baseline vs Final Prediction', fontsize=12, fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Plot 3: Correction vs Snapshot
axes[1,0].scatter(valid_snapshots, Delta_E, alpha=0.6, s=50, color='purple')
axes[1,0].axhline(Delta_E.mean(), color='red', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Snapshot ID', fontsize=11)
axes[1,0].set_ylabel('Predicted ΔE (kcal/mol)', fontsize=11)
axes[1,0].set_title('ML Corrections Across Snapshots', fontsize=12, fontweight='bold')
axes[1,0].grid(alpha=0.3)

# Plot 4: Energy comparison
axes[1,1].bar(range(len(results)), E_MM, alpha=0.5, label='E_MM', color='orange')
axes[1,1].bar(range(len(results)), E_pred, alpha=0.5, label='E_predicted', color='blue')
axes[1,1].set_xlabel('Snapshot Index', fontsize=11)
axes[1,1].set_ylabel('Energy (kcal/mol)', fontsize=11)
axes[1,1].set_title('Energy Comparison: MM vs Predicted', fontsize=12, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plot_file = output_dir / "predictions_analysis.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved plots to: {plot_file}")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*70)
print("PREDICTION SUMMARY")
print("="*70)

print(f"\nMM Baseline:")
print(f"  Range: {E_MM.min():.2f} to {E_MM.max():.2f} kcal/mol")
print(f"  Mean: {E_MM.mean():.2f} ± {E_MM.std():.2f} kcal/mol")

print(f"\nML Corrections (ΔE):")
print(f"  Range: {Delta_E.min():.2f} to {Delta_E.max():.2f} kcal/mol")
print(f"  Mean: {Delta_E.mean():.2f} ± {Delta_E.std():.2f} kcal/mol")

print(f"\nFinal Predictions (MM + ΔE):")
print(f"  Range: {E_pred.min():.2f} to {E_pred.max():.2f} kcal/mol")
print(f"  Mean: {E_pred.mean():.2f} ± {E_pred.std():.2f} kcal/mol")

print(f"\nCaveats:")
print(f"  - Model trained on Zn-6W (6 waters)")
print(f"  - Applied to Zn-18W (18 waters)")
print(f"  - No QM reference to validate accuracy")
print(f"  - Predictions are extrapolations, not interpolations")

print("\n" + "="*70)
print("COMPLETE! Ready for presentation.")
print("="*70)