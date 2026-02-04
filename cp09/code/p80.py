from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "The movie was full of incomprehensibilities."
tokens = tokenizer.tokenize(text)
print(tokens)
