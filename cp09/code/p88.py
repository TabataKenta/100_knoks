from transformers import BertTokenizer ,  BertForSequenceClassification
import torch

# p87で保存したフォルダから読み込む
model = BertForSequenceClassification.from_pretrained("./model_sst2")
tokenizer = BertTokenizer.from_pretrained("./model_sst2")

text1 ="The movie was full of incomprehensibilities."
text2="The movie was full of fun."
text3="The movie was full of excitement."
text4="The movie was full of crap."
text5="The movie was full of rubbish."

with torch.no_grad():
    for text in [text1, text2, text3, text4, text5]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f"Text: '{text}' => Predicted class: {predicted_class}")
