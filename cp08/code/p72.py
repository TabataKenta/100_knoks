import torch
import torch.nn as nn

class LogisticRegressionCBoW(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        
        # 埋め込み行列をモデルのパラメータの一部として保持（ただし更新はしない）
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        # 300次元から1次元へ変換する線形層（これが重み w とバイアス b に相当）
        self.linear = nn.Linear(300, 1)

    def forward(self, input_ids):
        # 1. input_ids からベクトルを取得
        embedded_vectors = self.embeddings(input_ids)
        
        # 2. 平均ベクトルを計算
        mean_vector = torch.mean(embedded_vectors, dim=1)
        
        # 3. 線形層に通す
        logits = self.linear(mean_vector)
        
        # 4. シグモイド関数を適用して 0~1 にする
        probs = torch.sigmoid(logits)

        return probs


if __name__ == '__main__':
     # 1. 保存した材料の読み込み
    embedding_matrix = torch.load('embedding_matrix.pt')
    train_dataset = torch.load('train_dataset.pt')
    
    # 2. モデルの初期化
    model = LogisticRegressionCBoW(embedding_matrix)
    
    # 3. テスト：最初の1事例を入力してみる
    example = train_dataset[0]
    input_ids = example['input_ids'] # この時点では [単語数] という1次元
    
    # モデルは (batch_size, seq_len) を期待しているので、0番目に次元を追加して (1, seq_len) にする
    input_batch = input_ids.unsqueeze(0) 
    
    # 推論モードに設定
    model.eval()
    with torch.no_grad():
        prob = model(input_batch)
    
    print(f"テキスト: {example['text']}")
    print(f"ポジティブである確率: {prob.item():.4f}")
    print(f"正解ラベル: {example['label'].item()}")
    
