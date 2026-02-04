from transformers import BertTokenizer , BertModel
import torch
import torch.nn as nn
from torch.optim import AdamW
from p85 import train_dataset, dev_dataset 
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---  アーキテクチャの変更: 最大値プーリングを用いる分類モデル ---
class MaxPoolingClassifier(nn.Module):
    def __init__(self, bert_model_name):
        super(MaxPoolingClassifier, self).__init__()
        # BERT本体
        self.bert = BertModel.from_pretrained(bert_model_name)
        # ドロップアウト (過学習防止)
        self.dropout = nn.Dropout(0.1)
        # 分類器 (入力はBERTの隠れ層サイズ, 出力はラベル数)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2) # 768 -> 2

    def forward(self, input_ids, attention_mask):
        # BERTに入力
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # last_hidden_stateの形状: [バッチサイズ, 系列長, 隠れ層サイズ]
        last_hidden_state = outputs.last_hidden_state
        
        # 系列長（トークンが並んでいる方向, dim=1）に対して最大値を取る
        # .valuesで最大値のテンソルのみ取得
        max_pooled_output = torch.max(last_hidden_state, dim=1).values
        
        # ドロップアウトを適用
        dropped_output = self.dropout(max_pooled_output)
        
        # 分類器に通してロジットを得る
        logits = self.classifier(dropped_output)
        
        return logits

if __name__ == "__main__":
    # 設定
    BATCH_SIZE = 64
    EPOCHS = 3
    LR = 1e-5
    BERT_MODEL_NAME = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MaxPoolingClassifier(BERT_MODEL_NAME).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        model.train()
        total_loss = 0
        
        # tqdmを使って進捗バーを表示
        train_progress = tqdm(train_loader, desc="Training")
        
        for batch in train_progress:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 予測を取得
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 損失を計算・逆伝播・重み更新
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_progress.set_postfix(loss=total_loss / (train_progress.n + 1))
            
    # 評価
    print("Training complete.\nStarting evaluation...")  
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # 勾配計算をオフ
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 予測を取得
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 予測ラベルを決定 (最大値のインデックスを取る)
            predictions = torch.argmax(logits, dim=-1)
            
            # 正解数をカウント
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    # 正解率を表示
    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")
