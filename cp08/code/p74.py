from p72 import LogisticRegressionCBoW
import torch
import torch.nn as nn 

# --- 正解率を計算する関数 ---
def calculate_accuracy(model, dataset, device):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for example in dataset:
            input_ids = example['input_ids'].unsqueeze(0).to(device)
            label = example['label'].unsqueeze(0).to(device)
            
            outputs = model(input_ids)
            
            # 確率(outputs)から予測(0 or 1)を判定
            pred = 1 if outputs.item() >= 0.5 else 0
            
            # 正解ならカウント
            if pred == int(label.item()):
                correct += 1
                
    return correct / len(dataset)

if __name__ == '__main__':
    # 材料の読み込み
    embedding_matrix = torch.load('embedding_matrix.pt')
    dev_dataset = torch.load('dev_dataset.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegressionCBoW(embedding_matrix).to(device)

    # 重みのロード
    weights = torch.load('p73_model.pt')
    model.load_state_dict(weights)

    # --- 正解率の計算を実行 ---
    acc = calculate_accuracy(model, dev_dataset, device)
    
    print(f"学習済みモデルの開発セットにおける正解率: {acc:.4f}")
