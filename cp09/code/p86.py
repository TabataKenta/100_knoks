from p85 import train_dataset
from torch.utils.data import DataLoader

# p85.pyで定義したSSTDatasetクラスを使ってデータセットを作成済み
# 4事例ずつのミニバッチを作る設定
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

# 最初の1バッチ（4事例分）だけ取り出す
batch = next(iter(train_loader))

# 形状を表示
print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")
print(f"Batch labels: {batch['labels']}")
