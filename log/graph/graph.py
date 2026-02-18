import matplotlib.pyplot as plt
import json
import os

item_list = ['bread', 'cracker', 'jelly', 'pudding', 'soda_water', 'yogurt', 'yokan']
metrics_dir = "../json"

# 保存先ディレクトリの作成（なければ）
os.makedirs("all", exist_ok=True)
for name in item_list:
    os.makedirs(name, exist_ok=True)

# 各クラスの評価グラフ作成
for name in item_list:
    metrics_path = os.path.join(metrics_dir, f"{name}.json")

    if not os.path.exists(metrics_path):
        print(f"{metrics_path} does not exist, skipping.")
        continue

    with open(metrics_path, "r") as f:
        data = json.load(f)

    epochs = data["epochs"]
    pixel_auroc = data["pixel_auroc"]
    sample_auroc = data["sample_auroc"]
    pixel_aupro = data["pixel_aupro"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, pixel_auroc, label='Pixel AUROC', marker='o')
    plt.plot(epochs, sample_auroc, label='Sample AUROC', marker='s')
    plt.plot(epochs, pixel_aupro, label='Pixel AUPRO', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(name)
    plt.ylim(0.00, 1.00)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(name, f"{name}.png"))  # 保存先変更
    plt.close()

# 全クラス分の評価指標を集約
metrics_all = {
    "Pixel AUROC": {},
    "Sample AUROC": {},
    "Pixel AUPRO": {}
}

for name in item_list:
    metrics_path = os.path.join(metrics_dir, f"{name}.json")

    if not os.path.exists(metrics_path):
        continue

    with open(metrics_path, "r") as f:
        data = json.load(f)

    epochs = data["epochs"]
    metrics_all["Pixel AUROC"][name] = (epochs, data["pixel_auroc"])
    metrics_all["Sample AUROC"][name] = (epochs, data["sample_auroc"])
    metrics_all["Pixel AUPRO"][name] = (epochs, data["pixel_aupro"])

# 評価指標ごとの比較グラフ作成
for metric_name, item_data in metrics_all.items():
    plt.figure(figsize=(10, 6))
    for name, (epochs, values) in item_data.items():
        plt.plot(epochs, values, label=name, marker='o')

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f"{metric_name}")
    plt.ylim(0.00, 1.00)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("all", f"{metric_name.replace(' ', '_').lower()}.png")  # 保存先変更
    plt.savefig(out_path)
    plt.close()
