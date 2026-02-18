"""
94. チャットテンプレート
“What do you call a sweet eaten after dinner?”という問いかけに対する応答を生成するため、
チャットテンプレートを適用し、言語モデルに与えるべきプロンプトを作成せよ。
また、そのプロンプトに対する応答を生成し、表示せよ。
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
        {"role": "user", "content": "What do you call a sweet eaten after dinner?"}
    ]
    
    # テンプレートの適用
    # Qwenは最初から正しいテンプレートを持っているので、apply_chat_templateを呼ぶだけでOK
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False, # トークン化は後で行うため、Falseに設定
        # 生成プロンプトを追加するかどうか。
        # Trueに設定することで、モデルが応答を生成するためのプロンプトが追加される。
        add_generation_prompt = True 
        )
    
    # プロンプトの確認
    print(f"プロンプト：{prompt}")
    
    # 応答の生成
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, # inputsの内容を展開して渡す
            max_length=100, # 生成するテキストの最大長
            do_sample=True, # 確率に基づいてランダムにトークンを選択する
            temperature=0.7, # 温度パラメータ
            top_p=0.9, # nucleus samplingのパラメータ
            pad_token_id=tokenizer.eos_token_id # パディングトークンID
        )
        
    # 生成されたテキストのデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成された応答：{generated_text}")
    
if __name__ == "__main__":
    main()
    
