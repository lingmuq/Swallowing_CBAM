import os
import pandas as pd
import matplotlib.pyplot as plt


csv_path = "eval.csv"

df = pd.read_csv(csv_path)

save_dir = os.path.dirname(os.path.abspath(csv_path))

metrics = {
    "Pixel AUROC": "Pixel Auroc",
    "Sample AUROC": "Sample Auroc",
    "Pixel AUPRO": "Pixel Aupro",
}


for title, column in metrics.items():
    plt.figure(figsize=(8, 5))
    plt.bar(df["class"], df[column])
    plt.xlabel("Class")
    plt.ylabel(title)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=30)

    save_path = os.path.join(save_dir, f"{column}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()