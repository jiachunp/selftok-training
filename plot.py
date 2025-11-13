import os
import math
import matplotlib.pyplot as plt
from PIL import Image

# Path to your folder
folder = "/data/code/SelfTok-o-shuffle/outputs_imagenet1k_shuffle_26k_cfg7_dog"

# Get all PNG files
png_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])

# Load images
images = [Image.open(os.path.join(folder, f)) for f in png_files]

# Compute grid size
n = len(images)
cols = 4   # roughly square grid
rows = 2

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

# Flatten axes for easy iteration
axes = axes.flatten() if n > 1 else [axes]

# Plot each image
for i, img in enumerate(images):
    axes[i].imshow(img)
    axes[i].set_title(png_files[i], fontsize=8)
    axes[i].axis("off")

# Hide extra subplots (if grid not full)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig('dog_shuffle_26k.png')
