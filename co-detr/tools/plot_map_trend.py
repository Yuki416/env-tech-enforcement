import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 13))
mAP = [
    0.083, 0.066, 0.001, 0.160, 0.082, 0.159, 
    0.101, 0.136, 0.082, 0.151, 0.072, 0.058
]

plt.figure(figsize=(10, 6))

# Plot the mAP line
plt.plot(epochs, mAP, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# Highlight Epoch 3 (Crash)
plt.plot(3, 0.001, marker='o', color='red', markersize=12)
plt.annotate('Crash Point\n(Epoch 3: 0.001)', 
             xy=(3, 0.001), 
             xytext=(3, 0.03),
             arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=2),
             fontsize=12, color='red', ha='center')

# Highlight Epoch 4 (Peak)
plt.plot(4, 0.160, marker='o', color='green', markersize=12)
plt.annotate('Peak / Highest\n(Epoch 4: 0.160)', 
             xy=(4, 0.160), 
             xytext=(4, 0.18),
             arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=2),
             fontsize=12, color='green', ha='center')

# Formatting
plt.title('Validation mAP Trend (Epoch 1 - 12)', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('mAP (bbox)', fontsize=14)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)

# Set Y limit to give some headroom for annotations
plt.ylim(-0.01, 0.20)

plt.tight_layout()

out_path = '/workspace/project/results/chiayi_12e/mAP_trend.png'
plt.savefig(out_path, dpi=300)
print(f"Plot saved to: {out_path}")
