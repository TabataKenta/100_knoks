"""
93. パープレキシティ
適当な文を準備して、事前学習済み言語モデルでパープレキシティを測定せよ。例えば、
・The movie was full of surprises

・The movies were full of surprises

・The movie were full of surprises

・The movies was full of surprises
の4文に対して、パープレキシティを測定して観察せよ（最後の2つの文は故意に文法的な間違いを入れた）。
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from p92 import calculate_propbability

texts =[
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises"
]

# モデルとトークナイザーの準備
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# テキストをトークン化してテンソルに変換
inputs = []
for text in texts:
    input = tokenizer(text, return_tensors='pt', add_special_tokens=True).to(device)
    inputs.append(input)

model.eval()   
with torch.no_grad():
    for input in inputs:
        outputs = model(**input, labels=input['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)
        print(f"テキスト: {tokenizer.decode(input['input_ids'][0], skip_special_tokens=True)}")
        print(f"パープレキシティ: {perplexity.item():.2f}\n")
        
        
