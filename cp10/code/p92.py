"""
92. 予測されたテキストの確率を計算
“The movie was full of”に続くテキストを予測し、生成された各単語の尤度を表示せよ
（生成されるテキストが長いと出力が読みにくくなるので、適当な長さで生成を打ち切るとよい）
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text(model,tokenizer,text,max_length=30):
    """
    GPT-2モデルを使ってテキストを生成する関数
    input:
        model: 事前学習済みの言語モデル
        tokenizer: モデルに対応するトークナイザー
        text(str): 入力テキスト
        max_length(int): 生成するテキストの最大長
    output:
        generated_text：生成されたテキスト
     """
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # テキストをトークン化してテンソルに変換
    inputs = tokenizer(text, return_tensors='pt').to(device)
    
    # テキストを生成するための共通の設定
    kwargs = {
        **inputs, # **inputs で input_ids と attention_mask を両方渡す
        'max_length': max_length, # 生成するテキストの最大長
        'pad_token_id': tokenizer.eos_token_id, # パディングトークンIDを指定する。GPT-2はパディングトークンを持たないため、EOSトークンを使用する。
        'do_sample': True, # 確率に基づいてランダムにトークンを選択する。
        'temperature': 1.0, # 温度パラメータ（1.0はデフォルトで、値が大きいほどランダム性が増す）
    }
    
    output_ids = model.generate(**kwargs)
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def calculate_propbability(model,tokenizer,text):
    """
    GPT-2モデルを使ってテキストの確率を計算する関数
    input:
        model: 事前学習済みの言語モデル
        tokenizer: モデルに対応するトークナイザー
        text(str): 入力テキスト
    output:
        token_probs：生成されたテキストの各トークンとその確率のリスト
     """
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # テキストをトークン化してテンソルに変換
    inputs = tokenizer(text, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 各トークンの確率を計算するために、softmax関数を適用する。
        probs = torch.softmax(logits, dim=-1)
        
        # input_ids の形状は (バッチサイズ, シーケンス長) で、probs の形状は (バッチサイズ, シーケンス長, 語彙数) 
        # トークンごとの確率を取得するために、入力のトークンIDを使用して、対応する確率を抽出する。
        input_ids = inputs['input_ids']
        token_probs = []
        for i in range(1, input_ids.shape[1]): # 最初の単語は予測ではないので 1 から始める
            # 「i-1」番目の位置のロジット（または確率）を見る
            prev_probs = probs[0, i-1, :]
            
            # 「i」番目の単語のIDを取得
            token_id = input_ids[0, i].item()
            
            # ズレを修正して確率を抽出
            prob = prev_probs[token_id].item()
            
            token_probs.append((tokenizer.decode([token_id]), prob))
            
    return token_probs

if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    text = "The movie was full of"

    # 1.テキスト生成
    generated_text = generate_text(model, tokenizer, text, max_length=30)
    print("生成されたテキスト:", generated_text)

    # 2.各トークンの確率を計算
    token_probs = calculate_propbability(model, tokenizer, generated_text)
    print("\n生成されたテキストの各トークンとその確率：")
    for token, prob in token_probs:
        print(f"トークン: {token}, 尤度: {prob:.6f}")
