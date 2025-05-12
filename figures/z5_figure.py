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
z5_models = [
    "R1 Baseline$^1$",
    "R1 MI",
    "Qwen Baseline",
    "Qwen MI",
    "Llama Baseline",
    "Llama MI"
]

z5_pass1      = [26, 39, 51, 73, 63, 75]
z5_passk      = [55, 80, 83, 87, 88, 86]
z5_pluralityk = [36, 44, 71, 74, 75, 78]
z5_consensusk = [15, 23, 56, 69, 63, 75]

# Adjust for incomplete data
z5_pass1[0] *= 100 / 78
z5_passk[0] *= 100 / 78
z5_pluralityk[0] *= 100 / 78
z5_consensusk[0] *= 100 / 78

fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

width = 0.18
indices = np.arange(len(z5_models))

# Plot data
bars1 = ax1.bar(indices - 1.5*width, z5_pass1, width, label='pass@1', color=colors['pass@1'])
bars2 = ax1.bar(indices - 0.5*width, z5_passk, width, label='pass@5', color=colors['pass@k'])
bars3 = ax1.bar(indices + 0.5*width, z5_pluralityk, width, label='plurality@5', color=colors['plurality@k'])
bars4 = ax1.bar(indices + 1.5*width, z5_consensusk, width, label='consensus@5', color=colors['consensus@k'])

add_value_labels(ax1, bars1)
add_value_labels(ax1, bars2)
add_value_labels(ax1, bars3)
add_value_labels(ax1, bars4)

ax1.set_xticks(indices)
ax1.set_xticklabels(z5_models)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_ylabel('GSM8K Performance', fontsize=12)
ax1.set_ylim(0, 100)
ax1.legend(loc='lower center', fontsize=10, ncol=4, bbox_to_anchor=(0.5, -0.18), frameon=False)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("model_performance_z5.png", dpi=300)
