from transformers import BertTokenizer ,  BertForSequenceClassification
import torch
from torch.optim import AdamW
from p85 import train_dataset, dev_dataset 
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, train_loader, dev_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        train_progress = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in train_progress:
            # 勾配の初期化
            self.optimizer.zero_grad()
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix(loss=total_loss / (train_progress.n + 1))
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        return total_correct / total_samples
        
    def save_model(self, output_dir, tokenizer):
        self.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
        
if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 3
    LR = 1e-5
    BERT_MODEL_NAME = 'bert-base-uncased'
    OUTPUT_DIR = "./model_sst2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データローダーの作成
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデルとオプティマイザの準備
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    
    # Trainerクラスを使って学習
    trainer = ModelTrainer(model, train_loader, dev_loader, optimizer, device)
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = trainer.train_epoch()
        print(f"Training Loss: {train_loss:.4f}")
        accuracy = trainer.evaluate()
        print(f"Validation Accuracy: {accuracy:.4f}")
        
    # モデルとトークナイザの保存
    trainer.save_model(OUTPUT_DIR, tokenizer)
