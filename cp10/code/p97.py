"""
97. 埋め込みに基づく感情分析
事前学習済み言語モデルでテキストをベクトルで表現（エンコード）し、
そのベクトルにフィードフォワード層を通すことで極性ラベルを予測するモデルを学習せよ。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel  # AutoModelForCausalLM ではなく AutoModel を使用
from datasets import load_dataset
from tqdm.auto import tqdm # 進捗表示のために tqdm をインポート

# 感情分析のための新しい分類モデルを定義するクラス
class SentimentClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 事前学習済みモデルをエンコーダ（特徴抽出器）として使用
        self.encoder = base_model
        # 事前学習済みモデルの隠れ層のサイズを取得（例: 768）
        hidden_size = base_model.config.hidden_size
        # 隠れ層のサイズのベクトルを受け取り、2つのクラス（ポジティブ/ネガティブ）へのスコアを出力する線形層
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        # エンコーダにトークンIDとアテンションマスクを渡して、各トークンのベクトル表現（隠れ状態）を得る
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # モデルの最終層の隠れ状態を取得。形状は (バッチサイズ, 系列長, 隠れ層サイズ)
        last_hidden_state = outputs.last_hidden_state  # (B, T, H)

        # アテンションマスクを使って、パディング部分を無視した平均プーリングを計算
        # 1. アテンションマスクを (B, T, 1) に拡張
        mask = attention_mask.unsqueeze(-1)
        # 2. 隠れ状態とマスクを掛け合わせる（パディング部分は0になる）
        masked_hidden_state = last_hidden_state * mask
        # 3. 系列方向（次元1）で合計をとり、有効なトークン数で割ることで平均を計算
        pooled_output = masked_hidden_state.sum(1) / mask.sum(1)

        # 平均化されたベクトルを分類層に通して、各クラスのスコア（ロジット）を得る
        logits = self.classifier(pooled_output)
        return logits


def main():
    # モデルとトークナイザーの準備
    # Instructモデルではなく、汎用的な特徴抽出に適したベースモデルを使用するのが一般的
    model_name = "Qwen/Qwen2.5-0.5B" # より軽量なモデルで試す
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # テキストをベクトル化する部分（モデルの本体）だけをロード
    base_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )


    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    # データセットの読み込み
    data_files = {
    "train": "SST-2/train.tsv",
    "validation": "SST-2/dev.tsv",
    }
    # 形状は、{"train": Dataset, "validation": Dataset, "test": Dataset} の辞書
    dataset = load_dataset("csv", data_files = data_files, delimiter="\t")

    train_data = dataset["train"]
    val_data = dataset["validation"]

    # 定義した分類器クラスをインスタンス化
    model = SentimentClassifier(base_model).to(device)

    # 最適化アルゴリズム（Adam）と損失関数（クロスエントロピー）を設定
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # DataLoaderに渡すためのカスタム関数。バッチ内のテキストをトークン化し、長さを揃える
    def collate(batch):
        # バッチからテキストとラベルをそれぞれ取り出す
        texts = [x["sentence"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch])
        # テキストをまとめてトークン化。padding=Trueでバッチ内の最長系列に合わせてパディング
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # ラベルも辞書に追加
        enc["labels"] = labels
        return enc

    # 学習用のDataLoaderを作成。shuffle=Trueでエポックごとにデータをシャッフル
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate)

    # --- 学習ループ ---
    model.train() # モデルを学習モードに設定
    for epoch in range(1): # 今回は1エポックだけ学習
        for batch in tqdm(train_loader, desc="Training"):
            # バッチ内の全データを指定のデバイス（GPUなど）に送る
            batch = {k: v.to(device) for k, v in batch.items()}
            # モデルにデータを入力し、ロジット（予測スコア）を得る
            logits = model(batch["input_ids"], batch["attention_mask"])
            # 予測ロジットと正解ラベルから損失を計算
            loss = criterion(logits, batch["labels"])

            # 勾配をリセット
            optimizer.zero_grad()
            # 損失を逆伝播させて勾配を計算
            loss.backward()
            # 計算した勾配に基づいてパラメータを更新
            optimizer.step()

    # --- 評価ループ ---
    model.eval() # モデルを評価モードに設定（Dropoutなどを無効化）
    correct = 0
    total = 0

    val_loader = DataLoader(val_data, batch_size=16, collate_fn=collate)

    with torch.no_grad(): # 勾配計算を無効にして、メモリ消費と計算時間を節約
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            # 最もスコアが高いクラスのインデックスを予測ラベルとする
            preds = torch.argmax(logits, dim=1)

            # 予測が正解と一致した数を加算
            correct += (preds == batch["labels"]).sum().item()
            # バッチサイズを合計に加算
            total += batch["labels"].size(0)

    print(f"Accuracy: {correct / total:.4f} ({correct} / {total})")


if __name__ == "__main__":
    main()
