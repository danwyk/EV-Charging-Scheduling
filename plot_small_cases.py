import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def symmetric_jitter(n, width):
    if n == 1:
        return np.array([0])
    else:
        return np.linspace(-width, width, n)

m_values = [2,3,4,5]
fig, axes = plt.subplots(2, 2, figsize=(16, 9)) 
axes = axes.flatten()  #
for idx, m in enumerate(m_values):
    ax = axes[idx]
    file_path = f"result/200_small/results_m{m}.csv"
    
    # read csv
    df = pd.read_csv(file_path)
    df_pivot = df.pivot(index="test_file", columns="algorithm", values="Cmax").reset_index()
    grouped = df_pivot.groupby("IP")

    # plot scatter
    instance_label_added = False
    for ip_val, group in grouped:
        ga_counts = group["GA"].value_counts().sort_index()

        for ga_val, count in ga_counts.items():
            jitters = symmetric_jitter(count, width=0.15)
            x_vals = ip_val + jitters
            y_vals = [ga_val] * count

            ax.scatter(
                x_vals,
                y_vals,
                color='black',
                marker='x',
                alpha=0.5,
                label='instance' if not instance_label_added else ""
            )
            instance_label_added = True

    # add lb line
    x_positions = sorted(grouped.groups.keys())
    ax.plot(x_positions, x_positions, color='green', linewidth=2, alpha=0.4, label='LB')

    ax.set_xticks(range(int(min(x_positions)), int(max(x_positions)) + 1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)) 
    ax.set_xlabel("Benchmark IP C_max value")
    ax.set_ylabel("GA C_max value")
    ax.set_title(f"Result on Solvable Small Instances (m={m}, n=10)")
    ax.grid(True)
    # add legend
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=1)

plt.tight_layout(rect=[0, 0.05, 1, 1])

save_path = "figure/200_small_result.png"
plt.savefig(save_path)
print(f"Figure saved to {save_path}")