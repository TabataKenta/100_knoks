from torch.utils.data import DataLoader
from p75 import collate
import tqdm
import torch
import torch.nn as nn
from p72 import LogisticRegressionCBoW

# p66.pyと同様の内容
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm.tqdm(loader): # dataset ではなく loader を回す
        optimizer.zero_grad()
        
        # すでにバッチ化されているので unsqueeze は不要
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # 1バッチ分の損失を合計（バッチサイズを考慮する必要がある点に注意）
        total_loss += loss.item() * inputs.size(0)
        
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval() # モデルを「評価モード」に設定
    total_loss = 0
    correct = 0
    
    with torch.no_grad(): # 評価時は勾配計算を無効化
        for batch in loader:
            # すでにバッチ化されているので unsqueeze は不要
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            # 予測と誤差の計算のみを行う
            outputs = model(inputs)
            
            # 0.5をしきい値にして、0か1の判定を下す
            # outputs は [バッチサイズ, 1] という形なので、それに対応する判定を作る
            preds = (outputs >= 0.5).float() 
            # 正解ラベル labels と一致している数を数える
            correct += (preds == labels).sum().item()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    
    accuracy = correct / len(loader.dataset)
    
    return total_loss / len(loader.dataset), accuracy

if __name__ == '__main__':
    embedding_matrix = torch.load('embedding_matrix.pt')
    train_dataset = torch.load('train_dataset.pt')
    dev_dataset = torch.load('dev_dataset.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LogisticRegressionCBoW(embedding_matrix).to(device)
    
    # batch_size=4 で、データをまとめる時に collate 関数を使う設定
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=collate)
    
    # 損失関数とオプティマイザの準備
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 学習率は適宜調整

    #  学習実行
    for epoch in range(10):
        # 学習
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # 評価
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Dev Loss = {dev_loss:.4f}, Dev Acc = {dev_acc:.4f}")
        
        dev_msg = f"Epoch {epoch+1}: Dev Loss = {dev_loss:.4f}, Dev Acc = {dev_acc:.4f}"
        with open('p76_result.txt', 'a') as f:
            f.write(dev_msg + '\n')
