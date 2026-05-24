import os
import pandas as pd
import matplotlib.pyplot as plt

# Create results directory
os.makedirs("results", exist_ok=True)

# Load CSV
df = pd.read_csv("results/metrics.csv")

# Rename models for cleaner display
model_map = {
    "AutoGluon_Chain": "AutoGluon",
    "XGBoost_Optimized": "XGBoost",
    "PyTorch_NN": "NN"
}

df["Model"] = df["Model"].map(model_map)

# Targets
targets = df["Target"].unique()

# Create 3x4 grid with more spacing
fig, axes = plt.subplots(
    3,
    4,
    figsize=(24, 14)
)

axes = axes.flatten()

# Plot each target
for i, target in enumerate(targets):
    ax = axes[i]

    subset = (
        df[df["Target"] == target]
        .sort_values("F3_Score", ascending=False)
    )

    bars = ax.bar(
        subset["Model"],
        subset["F3_Score"]
    )

    # Add values on top
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            fontsize=9
        )

    ax.set_title(
        target,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel("F3 Score")

    ax.set_ylim(0, 1)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

# Remove unused axes
for j in range(len(targets), len(axes)):
    fig.delaxes(axes[j])

# Global title
fig.suptitle(
    "Model Comparison by Target (F3 Score)",
    fontsize=20,
    fontweight="bold"
)

# Better spacing
plt.subplots_adjust(
    hspace=0.45,
    wspace=0.30,
    top=0.92
)

# Save figure
plt.savefig(
    "results/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()