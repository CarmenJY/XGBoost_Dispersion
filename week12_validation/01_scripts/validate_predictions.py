"""
Week 12: Validate Zn-18W predictions against QM reference energies
Run from DISPML directory: python week12_validation/01_scripts/validate_predictions.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("WEEK 12: ZN-18W PREDICTION VALIDATION")
print("="*70)

# ============================================================
# STEP 1: Load QM Reference Energies
# ============================================================
print("\n[1/5] Loading QM reference energies...")

qm_data = {}
with open("week12_validation/02_raw_data/zn18w_qm_energies.txt") as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        snapshot = int(parts[0])
        e_qm = float(parts[1])
        qm_data[snapshot] = e_qm

print(f"✓ Loaded QM energies for {len(qm_data)} snapshots")
print(f"  Range: {min(qm_data.values()):.2f} to {max(qm_data.values()):.2f} kcal/mol")
print(f"  Mean: {np.mean(list(qm_data.values())):.2f} kcal/mol")

# ============================================================
# STEP 2: Load Week 11 Predictions
# ============================================================
print("\n[2/5] Loading Week 11 predictions...")

with open("week11_zn18w_test/04_predictions/predictions.json") as f:
    predictions = json.load(f)

# Align data
snapshots = []
E_QM = []
E_MM = []
E_predicted = []
Delta_E_ML = []

for pred in predictions:
    snapshot = pred['snapshot']
    if snapshot in qm_data:
        snapshots.append(snapshot)
        E_QM.append(qm_data[snapshot])
        E_MM.append(pred['E_MM'])
        E_predicted.append(pred['E_predicted'])
        Delta_E_ML.append(pred['Delta_E_ML'])

E_QM = np.array(E_QM)
E_MM = np.array(E_MM)
E_predicted = np.array(E_predicted)
Delta_E_ML = np.array(Delta_E_ML)

print(f"✓ Aligned {len(snapshots)} snapshots with both QM and predictions")

# ============================================================
# STEP 3: Calculate Errors
# ============================================================
print("\n[3/5] Calculating prediction errors...")

# Baseline error (MM only)
rmse_baseline = np.sqrt(mean_squared_error(E_QM, E_MM))
mae_baseline = mean_absolute_error(E_QM, E_MM)
r2_baseline = r2_score(E_QM, E_MM)

# ML-corrected error
rmse_ml = np.sqrt(mean_squared_error(E_QM, E_predicted))
mae_ml = mean_absolute_error(E_QM, E_predicted)
r2_ml = r2_score(E_QM, E_predicted)

print(f"\nBaseline (MM only):")
print(f"  RMSE: {rmse_baseline:.3f} kcal/mol")
print(f"  MAE:  {mae_baseline:.3f} kcal/mol")
print(f"  R²:   {r2_baseline:.4f}")

print(f"\nML-Corrected (MM + ΔML):")
print(f"  RMSE: {rmse_ml:.3f} kcal/mol")
print(f"  MAE:  {mae_ml:.3f} kcal/mol")
print(f"  R²:   {r2_ml:.4f}")

improvement = (rmse_baseline - rmse_ml) / rmse_baseline * 100
print(f"\nImprovement: {improvement:.1f}% error reduction")

# Actual vs predicted residuals
actual_residual = E_QM - E_MM
predicted_residual = Delta_E_ML

residual_error = predicted_residual - actual_residual
rmse_residual = np.sqrt(mean_squared_error(actual_residual, predicted_residual))

print(f"\nResidual Prediction:")
print(f"  Actual ΔE (QM-MM): {actual_residual.mean():.2f} ± {actual_residual.std():.2f}")
print(f"  Predicted ΔE (ML): {predicted_residual.mean():.2f} ± {predicted_residual.std():.2f}")
print(f"  RMSE of residual prediction: {rmse_residual:.3f} kcal/mol")

# ============================================================
# STEP 4: Create Comprehensive Visualizations
# ============================================================
print("\n[4/5] Creating validation plots...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: MM vs QM
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(E_MM, E_QM, alpha=0.6, s=80, edgecolors='k', linewidth=0.5, color='orange')
min_val = min(E_MM.min(), E_QM.min())
max_val = max(E_MM.max(), E_QM.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax1.set_xlabel('E_MM (kcal/mol)', fontsize=12)
ax1.set_ylabel('E_QM (kcal/mol)', fontsize=12)
ax1.set_title(f'Baseline: RMSE = {rmse_baseline:.2f} kcal/mol', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: MM+ML vs QM
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(E_predicted, E_QM, alpha=0.6, s=80, edgecolors='k', linewidth=0.5, color='blue')
min_val = min(E_predicted.min(), E_QM.min())
max_val = max(E_predicted.max(), E_QM.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax2.set_xlabel('E_predicted (MM + ML) (kcal/mol)', fontsize=12)
ax2.set_ylabel('E_QM (kcal/mol)', fontsize=12)
ax2.set_title(f'With ML: RMSE = {rmse_ml:.2f} kcal/mol', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

# Plot 3: Error distributions
ax3 = plt.subplot(2, 3, 3)
errors_baseline = E_MM - E_QM
errors_ml = E_predicted - E_QM
ax3.hist(errors_baseline, bins=20, alpha=0.5, label='MM only', color='orange', edgecolor='black')
ax3.hist(errors_ml, bins=20, alpha=0.5, label='MM + ML', color='blue', edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Prediction Error (kcal/mol)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Error Distributions', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3)

# Plot 4: Residual prediction
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(actual_residual, predicted_residual, alpha=0.6, s=80, 
           edgecolors='k', linewidth=0.5, color='green')
min_val = min(actual_residual.min(), predicted_residual.min())
max_val = max(actual_residual.max(), predicted_residual.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax4.set_xlabel('Actual ΔE (QM - MM) (kcal/mol)', fontsize=12)
ax4.set_ylabel('Predicted ΔE (ML) (kcal/mol)', fontsize=12)
ax4.set_title(f'Residual Prediction: RMSE = {rmse_residual:.2f} kcal/mol', 
             fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# Plot 5: Error vs snapshot
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(snapshots, errors_ml, alpha=0.6, s=80, color='purple', edgecolors='k', linewidth=0.5)
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.axhline(errors_ml.mean(), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {errors_ml.mean():.2f}')
ax5.set_xlabel('Snapshot ID', fontsize=12)
ax5.set_ylabel('Prediction Error (kcal/mol)', fontsize=12)
ax5.set_title('ML Prediction Errors Across Snapshots', fontsize=13, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(alpha=0.3)

# Plot 6: Improvement summary
ax6 = plt.subplot(2, 3, 6)
categories = ['RMSE', 'MAE']
baseline_vals = [rmse_baseline, mae_baseline]
ml_vals = [rmse_ml, mae_ml]
x = np.arange(len(categories))
width = 0.35
bars1 = ax6.bar(x - width/2, baseline_vals, width, label='MM only', 
               color='orange', alpha=0.7, edgecolor='black')
bars2 = ax6.bar(x + width/2, ml_vals, width, label='MM + ML', 
               color='blue', alpha=0.7, edgecolor='black')
ax6.set_ylabel('Error (kcal/mol)', fontsize=12)
ax6.set_title('Performance Comparison', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(categories)
ax6.legend(fontsize=11)
ax6.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
output_dir = Path("week12_validation/03_analysis")
output_dir.mkdir(parents=True, exist_ok=True)
plot_file = output_dir / "validation_results.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot_file}")

# ============================================================
# STEP 5: Save Results
# ============================================================
print("\n[5/5] Saving validation results...")

results = {
    'baseline': {
        'rmse': float(rmse_baseline),
        'mae': float(mae_baseline),
        'r2': float(r2_baseline)
    },
    'ml_corrected': {
        'rmse': float(rmse_ml),
        'mae': float(mae_ml),
        'r2': float(r2_ml)
    },
    'improvement': {
        'rmse_reduction_pct': float(improvement),
        'absolute_improvement': float(rmse_baseline - rmse_ml)
    },
    'residual_prediction': {
        'rmse': float(rmse_residual),
        'actual_mean': float(actual_residual.mean()),
        'predicted_mean': float(predicted_residual.mean())
    }
}

results_file = output_dir / "validation_metrics.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save detailed comparison table
table_file = output_dir / "comparison_table.txt"
with open(table_file, 'w') as f:
    f.write("# Zn-18W Validation: QM vs Predictions\n")
    f.write("# Snapshot | E_QM | E_MM | E_pred | Error_MM | Error_ML\n")
    f.write("#" + "-"*70 + "\n")
    for i, snap in enumerate(snapshots):
        err_mm = E_MM[i] - E_QM[i]
        err_ml = E_predicted[i] - E_QM[i]
        f.write(f"{snap:3d} {E_QM[i]:10.2f} {E_MM[i]:10.2f} {E_predicted[i]:10.2f} "
            f"{err_mm:10.2f} {err_ml:10.2f}\n")

print(f"✓ Saved: {results_file}")
print(f"✓ Saved: {table_file}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("VALIDATION COMPLETE!")
print("="*70)

print(f"\n📊 Baseline Performance (MM only):")
print(f"   RMSE: {rmse_baseline:.3f} kcal/mol")
print(f"   MAE:  {mae_baseline:.3f} kcal/mol")

print(f"\n📊 ML-Corrected Performance:")
print(f"   RMSE: {rmse_ml:.3f} kcal/mol")
print(f"   MAE:  {mae_ml:.3f} kcal/mol")

print(f"\n✅ Improvement: {improvement:.1f}% error reduction")

if rmse_ml < 2.0:
    print("\n🎉 SUCCESS: Model achieves near-chemical-accuracy on Zn-18W!")
else:
    print("\n⚠️  Model performance on Zn-18W is acceptable but not chemical accuracy.")

print("\n" + "="*70)