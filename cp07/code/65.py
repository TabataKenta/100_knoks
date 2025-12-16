import joblib

# 62.pyで保存したモデルとベクトル化器の読み込み
model = joblib.load('62model.joblib')
vectorizer = joblib.load('62vectorizer.joblib')

def posi_nega_predict(text):
    # テキストの前処理
    text_list = text.split()
    seen = set()
    feature = {}
    for i in range(len(text_list)):
        if text_list[i] not in seen:
            seen.add(text_list[i])
            feature[text_list[i]] = 1
        else:
            feature[text_list[i]] += 1
        
    # 特徴量のベクトル化
    feature_vector = vectorizer.transform([feature])
    
    # 予測
    predicted_label = model.predict(feature_vector)
    
    return predicted_label[0]

# 使用例
input_text = "the worst movie I ‘ve ever seen"
print(posi_nega_predict(input_text))