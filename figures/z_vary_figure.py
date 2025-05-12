import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])

colors = {
    'pass@1': '#1f77b4',
    'pass@k': '#ff7f0e',
    'plurality@k': '#2ca02c',
    'consensus@k': '#d62728'
}

# Main data
z_models = [
    "Qwen Base ($z=5$)",
    "Qwen MI ($z=5$)",
    "Qwen Base ($z=10$)",
    "Qwen MI ($z=10$)",
    "Qwen Base ($z=20$)$^1$",
    "Qwen MI ($z=20$)"
]

z_pass1      = [51, 73, 63, 58, 30*2, 56]
z_passk      = [83, 87, 92, 89, 49*2, 97]
z_pluralityk = [71, 74, 75, 74, 41*2, 74]
z_consensusk = [56, 69, 62, 60, 28*2, 58]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

width = 0.18
indices = np.arange(len(z_models))

# Plot data
bars1 = ax1.bar(indices - 1.5*width, z_pass1, width, label='pass@1', color=colors['pass@1'])
bars2 = ax1.bar(indices - 0.5*width, z_passk, width, label='pass@k', color=colors['pass@k'])
bars3 = ax1.bar(indices + 0.5*width, z_pluralityk, width, label='plurality@k', color=colors['plurality@k'])
bars4 = ax1.bar(indices + 1.5*width, z_consensusk, width, label='consensus@k', color=colors['consensus@k'])

add_value_labels(ax1, bars1)
add_value_labels(ax1, bars2)
add_value_labels(ax1, bars3)
add_value_labels(ax1, bars4)

ax1.set_xticks(indices)
ax1.set_xticklabels(z_models)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_ylabel('GSM8K Performance', fontsize=12)
ax1.set_ylim(0, 100)
ax1.legend(loc='lower center', fontsize=10, ncol=4, bbox_to_anchor=(0.5, -0.18), frameon=False)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("model_performance_z_vary.png", dpi=300)
