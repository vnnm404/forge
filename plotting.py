import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

n = 4
with open('complex_vs_avg_degree.json') as f:
    data = json.load(f)
    N = data['N']
    x = data['Ds']
    y_labels = []
    ys = []
    for key in data["results"]:
        y_labels.append(key)
        ys.append(data["results"][key])
    
# Set the style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(12, 8))

# Use a colormap
colors = sns.color_palette("husl", n)

# Plot each line with a legend
for i, y in enumerate(ys):
    plt.plot(x, y, label=y_labels[i], color=colors[i], linewidth=2.5)

# Add titles and labels
plt.title("Multiple Line Plot", fontsize=20)
plt.xlabel("X-axis", fontsize=16)
plt.ylabel("Y-axis", fontsize=16)

# Customize the ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add legend
plt.legend(title="Legend", title_fontsize='13', fontsize='12')

# Show the plot

plt.savefig('complex_vs_avg_degree.png')
