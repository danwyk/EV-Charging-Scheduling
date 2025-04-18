import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

all_data = []
for m in range(4, 10):
    file_path = f"result/100_real/results_m{m}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

df_ga = df_all[df_all["algorithm"] == "GA"]
df_grouped = df_ga.groupby("n")["runtime_sec"].mean()

n_values = df_grouped.index.tolist()
x = np.arange(len(n_values))
width = 0.5

plt.figure(figsize=(12, 6))
bars = plt.bar(x, df_grouped.values, width, color='gray', alpha=0.8, label="GA")

plt.axhline(y=60, color='red', linestyle='--', linewidth=1.5, label="IP time limit (60s)")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height * 1.2,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=8
    )

plt.yscale("log")
plt.xticks(x, n_values)
plt.xlabel("Number of Jobs (n)")
plt.ylabel("Average GA Solve Time (s, log scale)")
plt.title("GA Average Runtime vs Number of Jobs (n) on Unsolvable Instances")
plt.grid(True, which="both", axis='y', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

save_path = "figure/100_real_runtime.png"
plt.savefig(save_path)
print(f"Figure saved to {save_path}")
plt.show()
