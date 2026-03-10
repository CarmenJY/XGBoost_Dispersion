"""
Train XGBoost model to predict induction energy residuals.
Run from DISPML directory: python week8_ml_training/01_scripts/train_model.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

print("="*70)
print("INDUCTION ML TRAINING - XGBoost Model")
print("="*70)

# ============================================================
# STEP 1: Load Residuals (from yesterday's analysis)
# ============================================================
print("\n[1/6] Loading residuals...")

data_file = Path("week8_ml_training/03_analysis/ml_training_data.npz")
data = np.load(data_file)

E_QM = data['E_QM']
E_MM = data['E_MM']
Delta_E = data['Delta_E']  # This is our target (y)

print(f"✓ Loaded {len(Delta_E)} residuals")
print(f"  Residual range: {Delta_E.min():.2f} to {Delta_E.max():.2f} kcal/mol")
print(f"  Residual mean: {Delta_E.mean():.2f} kcal/mol")
print(f"  Residual std: {Delta_E.std():.2f} kcal/mol")

# ============================================================
# STEP 2: Load Features (from Week 2 JSON files)
# ============================================================
print("\n[2/6] Loading geometric features from Week 2 data...")

feature_names = [
    'ZnO_min', 'ZnO_mean', 'ZnO_max', 'ZnO_std',
    'mean_invR2', 'mean_invR3', 'mean_invR4',
    'coord_ZnO_le_2p3', 'coord_ZnO_le_2p5', 'coord_ZnO_le_3p0',
    'OO_mean', 'OO_min', 'OO_std',
    'HB_count'
]

features_list = []
for replica in range(1, 6):
    for frame in range(1, 101):
        json_file = Path(f"output/REPLICA-{replica}_{frame}.json")
        
        if not json_file.exists():
            print(f"⚠ Warning: Missing {json_file}")
            continue
            
        with open(json_file) as f:
            feats = json.load(f)
            features_list.append([feats[fn] for fn in feature_names])

X = np.array(features_list)
y = Delta_E

print(f"✓ Loaded features for {len(X)} conformers")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Features: {len(feature_names)}")

# ============================================================
# STEP 3: Train-Test Split
# ============================================================
print("\n[3/6] Splitting data into train/validation/test sets...")

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: 75% train, 25% val (of the 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"✓ Data split complete:")
print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================
# STEP 4: Train XGBoost Model
# ============================================================
print("\n[4/6] Training XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print("✓ Model training complete")

# ============================================================
# STEP 5: Evaluate Performance
# ============================================================
print("\n[5/6] Evaluating model performance...")

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

train_rmse, train_mae, train_r2 = compute_metrics(y_train, y_train_pred, "Train")
val_rmse, val_mae, val_r2 = compute_metrics(y_val, y_val_pred, "Validation")
test_rmse, test_mae, test_r2 = compute_metrics(y_test, y_test_pred, "Test")

# Feature importance
feature_importance = model.feature_importances_
importance_pairs = sorted(zip(feature_names, feature_importance), 
                         key=lambda x: x[1], reverse=True)

print(f"\n  Top 5 Most Important Features:")
for feat, imp in importance_pairs[:5]:
    print(f"    {feat:20s}: {imp:.4f}")

# ============================================================
# STEP 6: Visualize Results
# ============================================================
print("\n[6/6] Creating visualization...")

fig = plt.figure(figsize=(14, 10))

# Plot 1: Predicted vs Actual (Test Set)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
min_val, max_val = y_test.min(), y_test.max()
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax1.set_xlabel('Actual ΔE (kcal/mol)', fontsize=11)
ax1.set_ylabel('Predicted ΔE (kcal/mol)', fontsize=11)
ax1.set_title(f'Test Set: RMSE = {test_rmse:.3f} kcal/mol', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Residual Errors (Test Set)
ax2 = plt.subplot(2, 3, 2)
errors = y_test_pred - y_test
ax2.hist(errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Mean: {errors.mean():.3f}')
ax2.set_xlabel('Prediction Error (kcal/mol)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Error Distribution (Test Set)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Feature Importance (Top 10)
ax3 = plt.subplot(2, 3, 3)
top_features = [x[0] for x in importance_pairs[:10]]
top_importance = [x[1] for x in importance_pairs[:10]]
bars = ax3.barh(range(len(top_features)), top_importance, color='steelblue', edgecolor='black')
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features, fontsize=9)
ax3.set_xlabel('Importance', fontsize=11)
ax3.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Learning Curves (All Sets)
ax4 = plt.subplot(2, 3, 4)
sets = ['Train', 'Val', 'Test']
rmses = [train_rmse, val_rmse, test_rmse]
colors = ['green', 'orange', 'red']
bars = ax4.bar(sets, rmses, color=colors, alpha=0.7, edgecolor='black')
for i, (s, r) in enumerate(zip(sets, rmses)):
    ax4.text(i, r + 0.02, f'{r:.3f}', ha='center', fontweight='bold')
ax4.set_ylabel('RMSE (kcal/mol)', fontsize=11)
ax4.set_title('Model Performance Across Sets', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Actual vs Predicted (All Data)
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(y_train, y_train_pred, alpha=0.3, s=20, label='Train', color='green')
ax5.scatter(y_val, y_val_pred, alpha=0.5, s=30, label='Val', color='orange')
ax5.scatter(y_test, y_test_pred, alpha=0.7, s=40, label='Test', color='red', edgecolors='k')
min_val = min(y_train.min(), y_val.min(), y_test.min())
max_val = max(y_train.max(), y_val.max(), y_test.max())
ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)
ax5.set_xlabel('Actual ΔE (kcal/mol)', fontsize=11)
ax5.set_ylabel('Predicted ΔE (kcal/mol)', fontsize=11)
ax5.set_title('All Data: Actual vs Predicted', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Final Prediction Quality
ax6 = plt.subplot(2, 3, 6)
# Compare E_QM with final prediction (E_MM + ML correction)
E_test_final = E_MM[len(E_MM)-len(y_test):] + y_test_pred
E_test_qm = E_QM[len(E_QM)-len(y_test):]
ax6.scatter(E_test_qm, E_test_final, alpha=0.6, s=50, color='purple', edgecolors='k', linewidth=0.5)
min_val, max_val = E_test_qm.min(), E_test_qm.max()
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
final_rmse = np.sqrt(mean_squared_error(E_test_qm, E_test_final))
ax6.set_xlabel('E_QM (SAPT) [kcal/mol]', fontsize=11)
ax6.set_ylabel('E_predicted (MM + ML) [kcal/mol]', fontsize=11)
ax6.set_title(f'Final Model: RMSE = {final_rmse:.3f} kcal/mol', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)

plt.tight_layout()
output_plot = Path("week8_ml_training/03_analysis/model_performance.png")
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_plot}")

# ============================================================
# STEP 7: Save Model
# ============================================================
print("\n[7/7] Saving trained model...")

model_file = Path("week8_ml_training/04_models/xgboost_model.json")
model_file.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(model_file))
print(f"✓ Saved: {model_file}")

# Save feature names
feature_file = Path("week8_ml_training/04_models/feature_names.txt")
with open(feature_file, 'w') as f:
    for fn in feature_names:
        f.write(f"{fn}\n")
print(f"✓ Saved: {feature_file}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n Final Test Set Performance:")
print(f"   RMSE: {test_rmse:.3f} kcal/mol")
print(f"   MAE:  {test_mae:.3f} kcal/mol")
print(f"   R²:   {test_r2:.4f}")

print(f"\n Final Prediction (E_MM + ML):")
print(f"   RMSE vs E_QM: {final_rmse:.3f} kcal/mol")

print(f"\n Model saved to: {model_file}")
print(f" Plots saved to: {output_plot}")

print("\n" + "="*70)
print("Ready for presentation!")
print("="*70)