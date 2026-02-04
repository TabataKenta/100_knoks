from transformers import BertTokenizer , BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# BERTモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 最終層の埋め込みベクトルの平均を取得する関数
def get_sentence_mean_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 最終層の埋め込みベクトルの平均を返す
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()



def calcurate_similarity(text1, text2):
    # 各文の埋め込みを取得
    embedding1 = get_sentence_mean_embedding(text1)
    embedding2 = get_sentence_mean_embedding(text2)
    
    # コサイン類似度の計算
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

if __name__ == "__main__":
    text1 = "The movie was full of fun."
    text2 = "The movie was full of excitement."
    text3 = "The movie was full of crap."
    text4 = "The movie was full of rubbish."
    
    texts = [text1, text2, text3, text4]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = calcurate_similarity(texts[i], texts[j])
            print(f"Similarity between '{texts[i]}' and '{texts[j]}': {sim:.4f}")
