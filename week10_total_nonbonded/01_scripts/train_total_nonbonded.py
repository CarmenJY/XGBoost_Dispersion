"""
Week 10: Total Nonbonded Energy ML Training
Combines QM total energy with MM total energy (Nonbonded + C4)
Run from DISPML directory: python week10_total_nonbonded/01_scripts/train_total_nonbonded.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

print("="*70)
print("WEEK 10: TOTAL NONBONDED ENERGY ML TRAINING")
print("="*70)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[1/7] Loading energy data...")

data_dir = Path("week10_total_nonbonded/02_raw_data")

# Load Total QM energies
qm_data = []
with open(data_dir / "total_qm_energies.txt") as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        replica, frame, e_qm = int(parts[0]), int(parts[1]), float(parts[2])
        qm_data.append((replica, frame, e_qm))

E_QM_total = np.array([x[2] for x in qm_data])
print(f"✓ Loaded {len(E_QM_total)} Total QM energies")
print(f"  Range: {E_QM_total.min():.2f} to {E_QM_total.max():.2f} kcal/mol")
print(f"  Mean: {E_QM_total.mean():.2f} kcal/mol")

# Load Nonbonded MM energies (El + C12 + C6)
nonbonded_data = {}
with open(data_dir / "nonbonded_energies.txt") as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        replica, frame, e_nb = int(parts[0]), int(parts[1]), float(parts[2])
        key = (replica, frame)
        # Handle duplicates - keep first occurrence
        if key not in nonbonded_data:
            nonbonded_data[key] = e_nb

print(f"✓ Loaded {len(nonbonded_data)} Nonbonded energies (El+C12+C6)")

# Load C4 energies (from Week 8)
c4_data = {}
with open(data_dir / "c4_energies.txt") as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        replica, frame, e_c4 = int(parts[0]), int(parts[1]), float(parts[2])
        c4_data[(replica, frame)] = e_c4

print(f"✓ Loaded {len(c4_data)} C4 energies (induction)")

# ============================================================
# STEP 2: Combine MM Energies
# ============================================================
print("\n[2/7] Computing total MM energies...")

E_MM_total = []
aligned_qm = []
replica_frame_list = []

for replica, frame, e_qm in qm_data:
    key = (replica, frame)
    
    if key in nonbonded_data and key in c4_data:
        e_nb = nonbonded_data[key]
        e_c4 = c4_data[key]
        e_mm_total = e_nb + e_c4
        
        E_MM_total.append(e_mm_total)
        aligned_qm.append(e_qm)
        replica_frame_list.append(key)

E_MM_total = np.array(E_MM_total)
E_QM_total = np.array(aligned_qm)

print(f"✓ Successfully aligned {len(E_MM_total)} conformers")
print(f"  MM Total range: {E_MM_total.min():.2f} to {E_MM_total.max():.2f} kcal/mol")
print(f"  MM Total mean: {E_MM_total.mean():.2f} kcal/mol")

# Breakdown
e_nb_sample = nonbonded_data[replica_frame_list[0]]
e_c4_sample = c4_data[replica_frame_list[0]]
print(f"\n  Example breakdown (REPLICA-{replica_frame_list[0][0]}, frame {replica_frame_list[0][1]}):")
print(f"    Nonbonded (El+C12+C6): {e_nb_sample:.2f} kcal/mol")
print(f"    C4 (induction):        {e_c4_sample:.2f} kcal/mol")
print(f"    Total MM:              {e_nb_sample + e_c4_sample:.2f} kcal/mol")

# ============================================================
# STEP 3: Compute Residuals
# ============================================================
print("\n[3/7] Computing residuals...")

Delta_E = E_QM_total - E_MM_total
baseline_pct = abs(E_MM_total.mean() / E_QM_total.mean()) * 100

print(f"Residual (ΔE = E_QM - E_MM):")
print(f"  Min: {Delta_E.min():.2f} kcal/mol")
print(f"  Max: {Delta_E.max():.2f} kcal/mol")
print(f"  Mean: {Delta_E.mean():.2f} kcal/mol")
print(f"  Std: {Delta_E.std():.2f} kcal/mol")
print(f"\nBaseline Performance:")
print(f"  MM captures {baseline_pct:.1f}% of total QM energy")
print(f"  Residual is {abs(Delta_E.mean() / E_QM_total.mean()) * 100:.1f}% of total")

# ============================================================
# STEP 4: Visualize Energy Analysis
# ============================================================
print("\n[4/7] Creating energy analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Energy distributions
axes[0,0].hist(E_QM_total, bins=30, alpha=0.7, label='E_QM (SAPT)', color='blue')
axes[0,0].hist(E_MM_total, bins=30, alpha=0.7, label='E_MM (Nonbonded+C4)', color='orange')
axes[0,0].set_xlabel('Energy (kcal/mol)', fontsize=11)
axes[0,0].set_ylabel('Count', fontsize=11)
axes[0,0].legend()
axes[0,0].set_title('Total Energy Distributions', fontsize=12, fontweight='bold')
axes[0,0].grid(alpha=0.3)

# Plot 2: Residual distribution
axes[0,1].hist(Delta_E, bins=30, color='green', alpha=0.7)
axes[0,1].axvline(Delta_E.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {Delta_E.mean():.2f}')
axes[0,1].set_xlabel('Residual ΔE (kcal/mol)', fontsize=11)
axes[0,1].set_ylabel('Count', fontsize=11)
axes[0,1].legend()
axes[0,1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[0,1].grid(alpha=0.3)

# Plot 3: MM vs QM scatter
axes[1,0].scatter(E_MM_total, E_QM_total, alpha=0.5, s=20)
min_val = min(E_MM_total.min(), E_QM_total.min())
max_val = max(E_MM_total.max(), E_QM_total.max())
axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
axes[1,0].set_xlabel('E_MM (kcal/mol)', fontsize=11)
axes[1,0].set_ylabel('E_QM (kcal/mol)', fontsize=11)
axes[1,0].legend()
axes[1,0].set_title('MM vs QM Total Energies', fontsize=12, fontweight='bold')
axes[1,0].grid(alpha=0.3)

# Plot 4: Residual vs MM
axes[1,1].scatter(E_MM_total, Delta_E, alpha=0.5, s=20, color='purple')
axes[1,1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1,1].set_xlabel('E_MM (kcal/mol)', fontsize=11)
axes[1,1].set_ylabel('Residual ΔE (kcal/mol)', fontsize=11)
axes[1,1].set_title('Residual vs MM Energy', fontsize=12, fontweight='bold')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
output_dir = Path("week10_total_nonbonded/03_analysis")
output_dir.mkdir(parents=True, exist_ok=True)
energy_plot = output_dir / "energy_analysis.png"
plt.savefig(energy_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {energy_plot}")
plt.close()

# ============================================================
# STEP 5: Load Features from Week 2
# ============================================================
print("\n[5/7] Loading geometric features...")

feature_names = [
    'ZnO_min', 'ZnO_mean', 'ZnO_max', 'ZnO_std',
    'mean_invR2', 'mean_invR3', 'mean_invR4',
    'coord_ZnO_le_2p3', 'coord_ZnO_le_2p5', 'coord_ZnO_le_3p0',
    'OO_mean', 'OO_min', 'OO_std', 'HB_count'
]

features_list = []
for replica, frame in replica_frame_list:
    json_file = Path(f"output/REPLICA-{replica}_{frame}.json")
    
    if json_file.exists():
        with open(json_file) as f:
            feats = json.load(f)
            features_list.append([feats[fn] for fn in feature_names])
    else:
        print(f"⚠ Warning: Missing {json_file}")

X = np.array(features_list)
y = Delta_E

print(f"✓ Loaded features for {len(X)} conformers")
print(f"  Feature matrix shape: {X.shape}")

# ============================================================
# STEP 6: Train XGBoost Model
# ============================================================
print("\n[6/7] Training XGBoost model...")

# Train-test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"✓ Data split:")
print(f"  Training:   {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test:       {len(X_test)} samples")

# Train model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("✓ Model training complete")

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Metrics
def compute_metrics(y_true, y_pred, set_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n  {set_name}:")
    print(f"    RMSE: {rmse:.3f} kcal/mol")
    print(f"    MAE:  {mae:.3f} kcal/mol")
    print(f"    R²:   {r2:.4f}")
    return rmse, mae, r2

print("\n  Model Performance:")
train_rmse, train_mae, train_r2 = compute_metrics(y_train, y_train_pred, "Train")
val_rmse, val_mae, val_r2 = compute_metrics(y_val, y_val_pred, "Validation")
test_rmse, test_mae, test_r2 = compute_metrics(y_test, y_test_pred, "Test")

# Feature importance
importance_pairs = sorted(zip(feature_names, model.feature_importances_), 
                         key=lambda x: x[1], reverse=True)
print(f"\n  Top 5 Features:")
for feat, imp in importance_pairs[:5]:
    print(f"    {feat:20s}: {imp:.4f}")

# ============================================================
# STEP 7: Save Results
# ============================================================
print("\n[7/7] Saving results...")

# Save data
np.savez(output_dir / "ml_training_data.npz",
         E_QM=E_QM_total,
         E_MM=E_MM_total,
         Delta_E=Delta_E)

# Save model
model_dir = Path("week10_total_nonbonded/04_models")
model_dir.mkdir(parents=True, exist_ok=True)
model.save_model(str(model_dir / "xgboost_model.json"))

with open(model_dir / "feature_names.txt", 'w') as f:
    for fn in feature_names:
        f.write(f"{fn}\n")

print(f"✓ Saved model and data")

# Create performance plot
fig = plt.figure(figsize=(12, 8))

# Plot 1: Test predictions
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
min_val, max_val = y_test.min(), y_test.max()
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax1.set_xlabel('Actual ΔE (kcal/mol)')
ax1.set_ylabel('Predicted ΔE (kcal/mol)')
ax1.set_title(f'Test Set: RMSE = {test_rmse:.3f} kcal/mol', fontweight='bold')
ax1.grid(alpha=0.3)

# Plot 2: Error distribution
ax2 = plt.subplot(2, 2, 2)
errors = y_test_pred - y_test
ax2.hist(errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Prediction Error (kcal/mol)')
ax2.set_ylabel('Count')
ax2.set_title('Error Distribution')
ax2.grid(alpha=0.3)

# Plot 3: Feature importance
ax3 = plt.subplot(2, 2, 3)
top_features = [x[0] for x in importance_pairs[:10]]
top_importance = [x[1] for x in importance_pairs[:10]]
ax3.barh(range(len(top_features)), top_importance, color='steelblue', edgecolor='black')
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features, fontsize=9)
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Features', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Performance comparison
ax4 = plt.subplot(2, 2, 4)
sets = ['Train', 'Val', 'Test']
rmses = [train_rmse, val_rmse, test_rmse]
bars = ax4.bar(sets, rmses, color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
for i, r in enumerate(rmses):
    ax4.text(i, r + 0.02, f'{r:.3f}', ha='center', fontweight='bold')
ax4.set_ylabel('RMSE (kcal/mol)')
ax4.set_title('Performance Across Sets', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
perf_plot = output_dir / "model_performance.png"
plt.savefig(perf_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {perf_plot}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Baseline Performance:")
print(f"   MM captures: {baseline_pct:.1f}% of total QM energy")
print(f"   Mean residual: {Delta_E.mean():.2f} ± {Delta_E.std():.2f} kcal/mol")

print(f"\n📊 ML Model Performance (Test Set):")
print(f"   RMSE: {test_rmse:.3f} kcal/mol")
print(f"   MAE:  {test_mae:.3f} kcal/mol")
print(f"   R²:   {test_r2:.4f}")

print(f"\n✅ Files saved:")
print(f"   {energy_plot}")
print(f"   {perf_plot}")
print(f"   {model_dir / 'xgboost_model.json'}")

print("\n" + "="*70)
print("Ready for presentation!")
print("="*70)