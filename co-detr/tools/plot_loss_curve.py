import json
import glob
import matplotlib.pyplot as plt
import os

# Find the latest scalars.json
directory = '/workspace/project/results/chiayi_12e'
scalar_files = glob.glob(f'{directory}/*/vis_data/scalars.json')
if not scalar_files:
    print("No scalars.json found!")
    exit(1)

# Grab the most recent one
latest_scalar_file = max(scalar_files, key=os.path.getmtime)
print(f"Reading logs from: {latest_scalar_file}")

steps = []
losses = []

with open(latest_scalar_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if 'loss' in data and 'step' in data:
                steps.append(data['step'])
                losses.append(data['loss'])
        except json.JSONDecodeError:
            continue

if not steps:
    print("No 'loss' data found to plot.")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label='Training Loss', color='b', alpha=0.7)

# Add smoothing (optional)
# window_size = 10
# if len(losses) > window_size:
#     import numpy as np
#     smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
#     plt.plot(steps[window_size-1:], smoothed, label='Smoothed Loss', color='r')

plt.title('Training Loss Curve')
plt.xlabel('Step / Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

output_path = os.path.join(directory, 'custom_loss_curve.png')
plt.savefig(output_path, dpi=300)
print(f"Loss curve saved to: {output_path}")