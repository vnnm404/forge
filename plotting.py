import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

n = 2
with open('complex_vs_avg_degree.json') as f:
    data = json.load(f)
    # N = data['N']
    x = data['Ds']
    # x = data['Ns']
    y_labels = []
    line_types = []
    ys = []
    for key in data["results"]:
        print(key)        
        if "nodes" in key:
            continue
        if "full_complex" in key:
            y_labels.append("Full Complex")
            line_types.append("1")
        else:
            y_labels.append("Reduced Complex")
            line_types.append("2")
        
        ys.append(data["results"][key])
    
# Set the style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(8.2, 8.2))
plt.tight_layout()
# Use a colormap
colors = sns.color_palette("husl", n)

# create a color dictionary with the labels
color_dict = {
    'Full Complex': colors[0],
    'Reduced Complex': colors[1]
}
print(color_dict)

# create a line type dictionary with the labels
line_dict = dict(zip(line_types, ["-", "--"]))

# Plot each line with large width
for i, y in enumerate(ys):
    print(color_dict[y_labels[i]])
    y = [i/1e5 for i in y]
    sns.lineplot(x=x, y=y, label=y_labels[i], color=color_dict[y_labels[i]], linestyle=line_dict[line_types[i]], linewidth=3.5)


plt.legend(title="Complex Type", loc="upper left", fontsize=24)

# make the ticks and labels bigger
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel("")
plt.ylabel("")

plt.savefig('complex_vs_avg_degree.svg')
