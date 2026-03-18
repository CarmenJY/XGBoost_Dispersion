"""
Week 11: Compute RMSE table comparing baseline vs ML-corrected predictions
Run from DISPML directory: python week10_total_nonbonded/01_scripts/compute_rmse_table.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import json

print("="*70)
print("WEEK 11: RMSE COMPARISON TABLE")
print("="*70)

# Load saved data
data_file = Path("week10_total_nonbonded/03_analysis/ml_training_data.npz")
data = np.load(data_file)

E_QM = data['E_QM']
E_MM = data['E_MM']
Delta_E = data['Delta_E']

# Load features
feature_names = [
    'ZnO_min', 'ZnO_mean', 'ZnO_max', 'ZnO_std',
    'mean_invR2', 'mean_invR3', 'mean_invR4',
    'coord_ZnO_le_2p3', 'coord_ZnO_le_2p5', 'coord_ZnO_le_3p0',
    'OO_mean', 'OO_min', 'OO_std', 'HB_count'
]

# Reconstruct feature matrix (same order as training)
features_list = []
for i in range(len(E_QM)):
    replica = (i // 100) + 1
    frame = (i % 100) + 1
    json_file = Path(f"output/REPLICA-{replica}_{frame}.json")
    
    if json_file.exists():
        with open(json_file) as f:
            feats = json.load(f)
            features_list.append([feats[fn] for fn in feature_names])

X = np.array(features_list)
y = Delta_E

# Same split as training
X_temp, X_test, y_temp, y_test, E_QM_temp, E_QM_test, E_MM_temp, E_MM_test = train_test_split(
    X, y, E_QM, E_MM, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val, E_QM_train, E_QM_val, E_MM_train, E_MM_val = train_test_split(
    X_temp, y_temp, E_QM_temp, E_MM_temp, test_size=0.25, random_state=42
)

# Load trained model
model = xgb.XGBRegressor()
model.load_model("week10_total_nonbonded/04_models/xgboost_model.json")

# Predict corrections
delta_train_pred = model.predict(X_train)
delta_val_pred = model.predict(X_val)
delta_test_pred = model.predict(X_test)

# Final predictions: E_MM + ML_correction
E_pred_train = E_MM_train + delta_train_pred
E_pred_val = E_MM_val + delta_val_pred
E_pred_test = E_MM_test + delta_test_pred

# Compute RMSEs
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Row 1: Between MM and QM (baseline)
rmse_mm_qm_train = rmse(E_QM_train, E_MM_train)
rmse_mm_qm_val = rmse(E_QM_val, E_MM_val)
rmse_mm_qm_test = rmse(E_QM_test, E_MM_test)

# Row 2: Between MM+delta and QM (final)
rmse_final_train = rmse(E_QM_train, E_pred_train)
rmse_final_val = rmse(E_QM_val, E_pred_val)
rmse_final_test = rmse(E_QM_test, E_pred_test)

print("\n" + "="*70)
print("RMSE COMPARISON TABLE")
print("="*70)
print("\n| Whose RMSE | Training set | Valid set | Test set |")
print("|------------|--------------|-----------|----------|")
print(f"| Between MM and QM | {rmse_mm_qm_train:.3f} | {rmse_mm_qm_val:.3f} | {rmse_mm_qm_test:.3f} |")
print(f"| Between MM+ΔML and QM | {rmse_final_train:.3f} | {rmse_final_val:.3f} | {rmse_final_test:.3f} |")

print("\n" + "="*70)
print("IMPROVEMENT SUMMARY")
print("="*70)
print(f"\nBaseline (MM only):")
print(f"  Train: {rmse_mm_qm_train:.3f} kcal/mol")
print(f"  Val:   {rmse_mm_qm_val:.3f} kcal/mol")
print(f"  Test:  {rmse_mm_qm_test:.3f} kcal/mol")

print(f"\nWith ML Correction (MM + ΔML):")
print(f"  Train: {rmse_final_train:.3f} kcal/mol")
print(f"  Val:   {rmse_final_val:.3f} kcal/mol")
print(f"  Test:  {rmse_final_test:.3f} kcal/mol")

print(f"\nImprovement:")
print(f"  Train: {rmse_mm_qm_train - rmse_final_train:.3f} kcal/mol ({(1 - rmse_final_train/rmse_mm_qm_train)*100:.1f}% reduction)")
print(f"  Val:   {rmse_mm_qm_val - rmse_final_val:.3f} kcal/mol ({(1 - rmse_final_val/rmse_mm_qm_val)*100:.1f}% reduction)")
print(f"  Test:  {rmse_mm_qm_test - rmse_final_test:.3f} kcal/mol ({(1 - rmse_final_test/rmse_mm_qm_test)*100:.1f}% reduction)")

print("\n" + "="*70)

# Save for slides
output_file = Path("week10_total_nonbonded/03_analysis/rmse_table.txt")
with open(output_file, 'w') as f:
    f.write("RMSE COMPARISON TABLE\n")
    f.write("="*70 + "\n\n")
    f.write("| Whose RMSE | Training set | Valid set | Test set |\n")
    f.write("|------------|--------------|-----------|----------|\n")
    f.write(f"| Between MM and QM | {rmse_mm_qm_train:.3f} | {rmse_mm_qm_val:.3f} | {rmse_mm_qm_test:.3f} |\n")
    f.write(f"| Between MM+ΔML and QM | {rmse_final_train:.3f} | {rmse_final_val:.3f} | {rmse_final_test:.3f} |\n")

print(f"✓ Table saved to: {output_file}")