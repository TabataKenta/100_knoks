import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class SSTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # 初期化関数
        # texts: 入力文のリスト
        # labels: 対応するラベルのリスト
        # truncation=True: 長すぎる場合に切り詰める
        # padding=True: 短い場合にパディングを追加
        self.encodings = tokenizer(texts, truncation=True, padding=True) # テキストをトークン化
        self.labels = labels

    def __len__(self):
        # データセットのサイズを返す関数
        return len(self.labels)

    def __getitem__(self, idx):
        # idx番目のデータを返す関数
        # 1. 辞書から特定のインデックス(idx)のデータを取り出し、テンソルに変換
        input_ids = torch.tensor(self.encodings['input_ids'][idx])
        attention_mask = torch.tensor(self.encodings['attention_mask'][idx])
        label = torch.tensor(self.labels[idx])

        # 2. モデルに渡すための辞書形式にまとめる
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }
        
        return item
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
# データの読み込み
df_train = pd.read_csv("SST-2/train.tsv", delimiter="\t")
df_dev = pd.read_csv("SST-2/dev.tsv", delimiter="\t")
    
train_texts = df_train['sentence'].tolist()
train_labels = df_train['label'].tolist()
dev_texts = df_dev['sentence'].tolist()
dev_labels = df_dev['label'].tolist()
    
train_dataset = SSTDataset(train_texts, train_labels, tokenizer)
dev_dataset = SSTDataset(dev_texts, dev_labels, tokenizer)

if __name__ == "__main__":
    # 動作確認
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of development samples: {len(dev_dataset)}")
    
    sample1 = train_dataset[0]
    print("Sample data from training set:")
    print(f"Input IDs: {sample1['input_ids']}")
    print(f"Attention Mask: {sample1['attention_mask']}")
    print(f"Label: {sample1['labels']}")
    print(f"Decoded Text: {tokenizer.decode(sample1['input_ids'])}")
    
    sample2 = dev_dataset[0]
    print("Sample data from development set:")
    print(f"Input IDs: {sample2['input_ids']}")
    print(f"Attention Mask: {sample2['attention_mask']}")
    print(f"Label: {sample2['labels']}")
    print(f"Decoded Text: {tokenizer.decode(sample2['input_ids'])}")
    
