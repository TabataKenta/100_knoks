import joblib
import json

# 62.pyで保存したモデルとベクトル化器の読み込み
model = joblib.load('62model.joblib')
vectorizer = joblib.load('62vectorizer.joblib')

# 検証データの読み込み
with open("61_output_dev.json", 'r') as f:
    dev_data = json.load(f)

# 先頭データの取り出し
first_data = dev_data[0]
feature = first_data['feature'] # 特徴量の辞書
label = first_data['label']     # 正解ラベル

# 特徴量のベクトル化
# 注意: transformはリストを受け取るので、辞書1つでもリストに入れて渡す必要がある
feature_vector = vectorizer.transform([feature])

# 条件付き確率の計算
predicted_proba = model.predict_proba(feature_vector)

# 結果の表示
print(f"確率：{predicted_proba}")