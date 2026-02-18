"""
95. マルチターンのチャット
問題94で生成された応答に対して、追加で
”Please give me the plural form of the word with its spelling in reverse order.”
と問いかけたときの応答を生成・表示せよ。
また、その時に言語モデルに与えるプロンプトを確認せよ。
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # モデルとトークナイザーの準備
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # チャットテンプレートの作成
    messages = [
        {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
        {"role": "assistant", "content": "A dessert."},
        {"role": "user", "content": "Please give me the plural form of the word with its spelling in reverse order."}
    ]
    
    # テンプレートの適用
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False, # トークン化は後で行うため、Falseに設定
        add_generation_prompt = True # 生成プロンプトを追加するかどうか。Trueに設定することで、モデルが応答を生成するためのプロンプトが追加される。
        )
    
    # プロンプトの確認
    print(f"プロンプト：{prompt}")
    
    # 応答の生成
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, # inputsの内容を展開して渡す
            max_new_tokens=50, # 生成するテキストの最大長を新しいトークン数で指定
            do_sample=False, # 確率の高いトークンを選択する
            pad_token_id=tokenizer.eos_token_id # パディングトークンID
        )
        
    # 応答部分
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("生成された応答：", generated_text)

if __name__ == "__main__":
    main()
