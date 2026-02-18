"""
96. プロンプトによる感情分析
事前学習済み言語モデルで感情分析を行いたい。
テキストを含むプロンプトを事前学習済み言語モデルに与え、
（ファインチューニングは行わずに）テキストのポジネガを予測するという戦略で、
SST-2の開発データにおける正解率を測定せよ。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():
    # モデルとトークナイザーの準備
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # データセットの読み込み
    data_files = {
    "train": "SST-2/train.tsv",
    "validation": "SST-2/dev.tsv",
    }
    # 形状は、{"train": Dataset, "validation": Dataset, "test": Dataset} の辞書
    dataset = load_dataset("csv", data_files = data_files, delimiter="\t")
    
    # SST-2の開発データからテキストとラベルを抽出
    val_data = dataset["validation"]
    
    correct = 0
    total = 0
    for item in val_data:
        text = item["sentence"]
        label = item["label"]
        
        messages = [
            {"role": "system", "content": "You are a sentiment analyzer. Answer only 'positive' or 'negative'."},
            {"role": "user", "content": f"Sentence: {text}"}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        model.eval()
        with torch.no_grad():
            # outputs には、[入力のトークンID, 生成されたトークンID] が含まれる。
            # outputs[0] = [プロンプトのID列, 生成されたID列]
            outputs = model.generate(
                **inputs,
                max_new_tokens=5, # 生成するテキストの最大長を新しいトークン数で指定
                do_sample=False
            )
        
        # 入力プロンプトのトークン数
        input_length = inputs["input_ids"].shape[1]
    
        generated_ids = outputs[0][input_length:] # 生成された部分のトークンIDを抽出
        
        # 生成されたテキストをデコード
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
        
        # 予測されたラベルを抽出
        if "positive" in generated_text.lower():
            predicted_label = 1
        elif "negative" in generated_text.lower():
            predicted_label = 0
        else:
            predicted_label = -1  # 不明なラベルの場合
        
        if predicted_label == label:
            correct += 1
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")
    
if __name__ == "__main__":
    main()
