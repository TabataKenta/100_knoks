"""
98. ファインチューニング
問題96のプロンプトに対して、
正解の感情ラベルをテキストの応答として返すように事前学習済みモデルをファインチューニングせよ。
"""
"""
98. ファインチューニング
問題96のプロンプトに対して、正解の感情ラベルをテキストとして返すようにモデルを学習する。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # 1. モデルとトークナイザーの準備
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # パディング用トークンの設定（Qwenにパディング用設定がない場合の安全策）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. データセットの読み込み
    data_files = {
        "train": "SST-2/train.tsv",
        "validation": "SST-2/dev.tsv",
    }
    # ローカルのTSVファイルを読み込む
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    # 3. データをチャット形式に変換する関数
    def format_example(example):
        # 0/1のラベルを文字列に変換
        label_text = "positive" if example["label"] == 1 else "negative"

        # ファインチューニング用のメッセージ構成（System, User, Assistantの3つをセットにする）
        messages = [
            {"role": "system", "content": "You are a sentiment analyzer. Answer only 'positive' or 'negative'."},
            {"role": "user", "content": f"Sentence: {example['sentence']}"},
            {"role": "assistant", "content": label_text} # 正解（Assistantの返答）も含める
        ]

        # チャットテンプレートを適用して1本の文字列にする
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        return {"text": text}

    # 学習データ全体に適用
    train_dataset = train_data.map(format_example)

    # 4. バッチ作成用の関数（Collate function）
    def collate(batch):
        texts = [x["text"] for x in batch]
        # テキストをトークン化し、長さを揃える
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Causal LM（因果的言語モデル）の学習では、入力IDそのものがラベルになる
        # モデル内部で「次のトークン」を予測するように自動でずらして計算される
        enc["labels"] = enc["input_ids"].clone()
        
        # パディング部分は損失計算から除外するため、ラベルを -100 に書き換える（PyTorchの慣例）
        enc["labels"][enc["input_ids"] == tokenizer.pad_token_id] = -100
        
        return enc

    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate)

    # 5. 最適化アルゴリズムの設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 6. 学習ループ
    model.train()
    print("Starting training...")
    for epoch in range(1):
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # モデルの実行（labelsを渡すと自動でLossが計算される）
            outputs = model(**batch)
            loss = outputs.loss

            # 勾配の更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(train_loader):.4f}")

    # 7. 評価（ファインチューニング後の性能確認）
    model.eval()
    correct = 0
    total = 0

    print("Starting evaluation...")
    # 開発データから最初の100件などでテスト（全件だと時間がかかるため）
    test_subset = val_data.select(range(min(100, len(val_data))))

    for item in tqdm(test_subset):
        label = item["label"]

        # 評価時は Assistant の返答は含めない（add_generation_prompt=True）
        messages = [
            {"role": "system", "content": "You are a sentiment analyzer. Answer only 'positive' or 'negative'."},
            {"role": "user", "content": f"Sentence: {item['sentence']}"}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # モデルに応答を生成させる
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        # 生成された部分だけを抽出してデコード
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip().lower()

        # 予測の判定
        if "positive" in generated:
            pred = 1
        elif "negative" in generated:
            pred = 0
        else:
            pred = -1 # 判定不能

        if pred == label:
            correct += 1
        total += 1

    print(f"Validation Accuracy: {correct / total:.4f} ({correct} / {total})")

if __name__ == "__main__":
    main()
