import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, m in enumerate(range(4, 10)):  # m from 4 to 9
    ax = axes[idx]
    df = pd.read_csv(f"result/100_real/results_m{m}.csv") 

    df_pivot = df.pivot(index="test_file", columns="algorithm", values="Cmax").reset_index()
    df_meta = df.drop(columns=["algorithm", "Cmax"]).drop_duplicates()
    df_merged = pd.merge(df_pivot, df_meta, on="test_file")
    df_merged["gap_percent"] = (df_merged["GA"] - df_merged["IP"]) / df_merged["IP"] * 100

    grouped = df_merged.groupby("n")
    x_positions = list(grouped.groups.keys())
    box_data = [group["gap_percent"].values for _, group in grouped]

    box = ax.boxplot(
        box_data,
        positions=x_positions,
        patch_artist=True,
        widths=2.5,
        showfliers=True
    )

    for patch in box['boxes']:
        patch.set_facecolor('lightgray')
        patch.set_alpha(0.7)

    ax.set_title(f"m={m}")
    ax.set_xlabel("Number of Jobs (n)")
    ax.set_ylabel("GAP (%)")
    ax.grid(True, axis='y')

plt.suptitle("C_max GAP% on Unsolvable Instances (IP Limited Solve Time=60s)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  

save_path = "figure/10_real_combined_boxplot.png"
plt.savefig(save_path)
print(f"Combined figure saved to {save_path}")
plt.show()
