"""
90. 次単語予測
“The movie was full of”に続くトークン（トークン列ではなく一つのトークンであることに注意せよ）
として適切なもの上位10個と、その確率（尤度）を求めよ。
ただし、言語モデルへのプロンプトがどのようなトークン列に変換されたか、確認せよ。
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch


def predict_next_token(text,top_k=10):
    """
    GPT-2モデルを使って次のトークンを予測する関数
    input:
        text(str): 入力テキスト
        top_k(int): 上位k個のトークン
    output:
        result：トップk個のトークンとその確率のリスト
        input_tokens：トークンID列
     """
    # モデルとトークナイザーの準備
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # テキストをトークン化してテンソルに変換
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # モデルを評価モードに設定
    model.eval()
    with torch.no_grad():
        # model(input_ids):入力されたトークンIDをモデルに入れて、出力を得る。    
        outputs = model(input_ids)
        # outputs.logits:モデルの出力のロジットを取得する。
        # ロジットは、各トークンが次に来る確率の前の値で、softmax関数を通す前の値。
        logits = outputs.logits
        # logits.shape = (バッチサイズ, シーケンス長, 語彙数)
        # 最後のトークンのロジットを取得
        # :→全てのバッチ、-1→最後のトークン、:→全ての語彙
        last_logits = logits[:, -1, :]
        # 確率を計算
        probs = torch.softmax(last_logits, dim=-1)
        # 上位k個のトークンと確率を取得
        # torch.topk()は、テンソルの中から上位k個の値とそのインデックスを返す関数
        top_probs, top_indices = torch.topk(probs, k=top_k)
        # トークンを文字列に変換して返す
        result = []
        for i in range(top_k):
            # decoder.decode()は、トークンIDを文字列に変換する関数
            # top_indicaes:上位のトークンのインデックスを含むテンソルで、形状は(バッチサイズ, k)。
            # top_probs:上位k個のトークンの確率を含むテンソルで、形状は(バッチサイズ, k)
            # .item()でテンソル→数値
            token = tokenizer.decode([top_indices[0][i].item()])
            prob = top_probs[0][i].item()
            result.append((token, prob))
            
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return result, input_tokens

    
if __name__ == "__main__":
    text = "The movie was full of"
    print(f"入力テキスト: {text}")
    predictions, input_tokens = predict_next_token(text, top_k=10)
    print(f"トークンID列: {input_tokens}")
    
    print("\n上位10個のトークンとその確率:")
    for token, prob in predictions:
        print(f"トークン: '{token}', 確率: {prob:.6f}")
    
    

