from p72 import LogisticRegressionCBoW
import torch
import torch.nn as nn 
import tqdm

# --- 学習用の関数 ---
def train(model, dataset, criterion, optimizer, device):
    model.train() # モデルを「学習モード」に設定
    train_loss = 0
    
    for example in tqdm.tqdm(dataset, desc="Training"):
        # 前のデータの反省内容（勾配）をリセット
        optimizer.zero_grad()
        
        # データの次元調整：[単語数] -> [1, 単語数] (1件入りのバッチ形式にする)
        # かつ、計算デバイス（CPU/GPU）に転送
        input_ids = example['input_ids'].unsqueeze(0).to(device)
        label = example['label'].unsqueeze(0).to(device)
        
        # 順伝搬：今のモデルで予測確率を計算
        outputs = model(input_ids)
        
        # 損失計算：正解とのズレ（誤差）を計算
        loss = criterion(outputs, label)
        
        # 逆伝搬：誤差に基づいて「どの重みをどう直すべきか」を計算（分析）
        loss.backward()
        
        # 重みの更新：分析結果をもとに実際に重みを書き換える
        optimizer.step()
        
        # 誤差を記録（後で平均を出すため）
        train_loss += loss.item()
        
    return train_loss / len(dataset)

# --- 評価用の関数：（重みは更新しない） ---
def evaluate(model, dataset, criterion, device):
    model.eval() # モデルを「評価モード」に設定
    total_loss = 0
    
    with torch.no_grad(): # 評価時は勾配計算を無効化
        for example in dataset:
            # 学習時と同じく次元調整とデバイス転送
            input_ids = example['input_ids'].unsqueeze(0).to(device)
            label = example['label'].unsqueeze(0).to(device)
            
            # 予測と誤差の計算のみを行う
            outputs = model(input_ids)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            
    return total_loss / len(dataset) 
    
if __name__ == '__main__':
    # 1. 前の問題で保存した材料（行列・データセット）を読み込む
    embedding_matrix = torch.load('embedding_matrix.pt')
    train_dataset = torch.load('train_dataset.pt')
    dev_dataset = torch.load('dev_dataset.pt')

    # 2. 計算デバイスの決定（GPUがあればGPU、なければCPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. モデルの準備とデバイスへの転送
    model = LogisticRegressionCBoW(embedding_matrix).to(device)
    
    # 4. 損失関数（ズレの測定器）とオプティマイザ（重みの更新担当）の設定
    criterion = nn.BCELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 学習率0.01でSGDを使用
    
    # 5. 学習ループの実行
    epochs = 5
    for epoch in range(epochs):
        # 訓練データで学習を実行
        avg_train_loss = train(model, train_dataset, criterion, optimizer, device)
        
        # 検証データで実力を測定（モニタリング）
        avg_dev_loss = evaluate(model, dev_dataset, criterion, device)
        
        # 進捗を表示
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Dev Loss = {avg_dev_loss:.4f}")
        
    # モデルの保存
    torch.save(model.state_dict(), 'p73_model.pt')
