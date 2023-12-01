import numpy as np
import pandas as pd
import geopandas as gpd
import GiovConfig as config
from tqdm import tqdm
from GiovanniDataset import GiovanniDataset
import torch
from pathlib import Path
import imageio
import matplotlib.pyplot as plt

models = ['All', 'All - [Municipalities]', 'All - [Municipalities, precipitation]', 'All - [Municipalities, precipitation, tpi]',\
    'All - [Municipalities, precipitation, tpi, past_scores]', 'All - [Municipalities, precipitation, tpi, past_scores, night light]']

models = list(range(len(models)))
models = ['All', 'All1 - Municipalities', 'All2 - Precipitation', 'All3 - TPI', 'All4 - Accumulated Def', 'All5 - Night Light']
f1_scores = [0.6172, 0.5338, 0.9587, 0.2078, 0.1093, 0.2755]

# Create a figure with an initial size
scale = 3
fig, ax = plt.subplots(figsize=(4*scale, 3*scale))

# Create a bar plot
bars = ax.bar(models, f1_scores, color='blue', edgecolor='black', alpha=0.7)

fontsize = 13
# Add labels to each bar
for bar, name in zip(bars, models):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=fontsize)
    
# Add labels and title
ax.set_xlabel('Features used in each experiment', fontsize=(fontsize + 2))
ax.set_ylabel('F1-Scores', fontsize=(fontsize + 2))
ax.set_title('Feature Comparision', fontsize=(fontsize + 4))
ax.tick_params(axis='both', labelsize=fontsize - 1)  # Adjust the fontsize as needed


plt.tight_layout()
# Show the plot
plt.show()

# Save the figure with a resolution of 300 dpi
fig.savefig('feature_comparision.png', dpi=300)

# # Generate random colors for each point
# colors = np.random.rand(len(f1_scores), 3)
# print(colors)
# # Create a scatter plot
# plt.scatter(range(1, len(f1_scores) + 1), f1_scores, label='Feature Comparision', color=colors, marker='*')

# # Add legend with corresponding names
# for i, txt in enumerate(models):
#     plt.annotate(txt, (i + 1, f1_scores[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# # Add labels and title
# plt.xlabel('Models')
# plt.ylabel('F1-Scores')
# plt.title('Feature Comparision')

# # Show the plot
# plt.show()

# # Create a plot
# plt.plot(f1_scores, label='F1-scores')

# # Add legend with corresponding names
# plt.legend(labels=[f'{name}: {value}' for name, value in zip(models, f1_scores)])

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Float Values')
# plt.title('Plot of Float Values with Legend')

# # Show the plot
# plt.show()