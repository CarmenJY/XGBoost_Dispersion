"""
Split the 6-panel model_performance.png into individual panel images.
Run from DISPML directory: python week8_ml_training/01_scripts/split_panels.py
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

print("Splitting model_performance.png into individual panels...")

# Load the full image
img_path = Path("week8_ml_training/03_analysis/model_performance.png")
output_dir = Path("week8_ml_training/03_analysis/panels")
output_dir.mkdir(exist_ok=True)

img = mpimg.imread(img_path)

# Get image dimensions
height, width, _ = img.shape

# The plot is 2 rows × 3 columns
rows, cols = 2, 3

# Calculate panel dimensions (with some margin handling)
panel_height = height // rows
panel_width = width // cols

# Panel labels and descriptions
panels = {
    1: "predicted_vs_actual_test",
    2: "error_distribution",
    3: "feature_importance",
    4: "performance_across_sets",
    5: "all_data_predicted_vs_actual",
    6: "final_prediction_quality"
}

# Extract and save each panel
for panel_num, panel_name in panels.items():
    # Calculate position (1-indexed to 0-indexed)
    row = (panel_num - 1) // cols
    col = (panel_num - 1) % cols
    
    # Extract region
    y_start = row * panel_height
    y_end = (row + 1) * panel_height
    x_start = col * panel_width
    x_end = (col + 1) * panel_width
    
    panel_img = img[y_start:y_end, x_start:x_end]
    
    # Save individual panel
    output_file = output_dir / f"panel_{panel_num}_{panel_name}.png"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(panel_img)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"✓ Saved: {output_file}")

print(f"\n✓ All panels saved to: {output_dir}/")
print("\nPanel descriptions:")
print("  Panel 1: Predicted vs Actual (Test Set)")
print("  Panel 2: Error Distribution")
print("  Panel 3: Feature Importance (Top 10)")
print("  Panel 4: Performance Across Train/Val/Test")
print("  Panel 5: All Data (Train + Val + Test)")
print("  Panel 6: Final Prediction Quality (E_MM + ML vs E_QM)")