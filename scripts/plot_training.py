"""Generate training curves for README."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load training log
with open("outputs/training_log.json") as f:
    log = json.load(f)

epochs = [e["epoch"] for e in log]
losses = [e["train_loss"] for e in log]
r1 = [e["recall@1"] * 100 for e in log]
r4 = [e["recall@4"] * 100 for e in log]
r8 = [e["recall@8"] * 100 for e in log]
triplets = [e["train_triplets"] for e in log]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Colors
BLUE = '#4A90D9'
RED = '#E74C3C'
GREEN = '#2ECC71'
ORANGE = '#F39C12'
PURPLE = '#9B59B6'

# Plot 1: Loss curve
axes[0].plot(epochs, losses, color=BLUE, linewidth=2, marker='o', markersize=4)
axes[0].set_title('Training Loss (Triplet Margin)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, max(epochs))

# Plot 2: Recall@K curves
axes[1].plot(epochs, r1, color=RED, linewidth=2, marker='o', markersize=4, label='Recall@1')
axes[1].plot(epochs, r4, color=GREEN, linewidth=2, marker='s', markersize=4, label='Recall@4')
axes[1].plot(epochs, r8, color=PURPLE, linewidth=2, marker='^', markersize=4, label='Recall@8')
axes[1].axhline(y=max(r1), color=RED, linestyle='--', alpha=0.4, linewidth=1)
axes[1].set_title('Recall@K (Test Set)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Recall (%)', fontsize=11)
axes[1].legend(fontsize=10, loc='lower right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, max(epochs))
axes[1].set_ylim(40, 85)

# Plot 3: Triplets mined (shows model learning — fewer hard triplets over time)
axes[2].plot(epochs, [t/1000 for t in triplets], color=ORANGE, linewidth=2, marker='D', markersize=4)
axes[2].set_title('Semi-Hard Triplets Mined (K)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=11)
axes[2].set_ylabel('Triplets (thousands)', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(1, max(epochs))

plt.tight_layout()
os.makedirs("outputs/visualizations", exist_ok=True)
plt.savefig("outputs/visualizations/training_curves.png", dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Training curves saved to outputs/visualizations/training_curves.png")
