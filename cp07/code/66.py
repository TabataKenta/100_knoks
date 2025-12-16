import json
import joblib
from sklearn.metrics import confusion_matrix

# 62.pyで保存したモデルとベクトル化器の読み込み
model = joblib.load('62model.joblib')
vectorizer = joblib.load('62vectorizer.joblib')

# 検証データの読み込み
with open("61_output_dev.json", 'r') as f:
    dev_data = json.load(f)
    
x_dev_dicts = [] # 特徴量の辞書のリスト
y_dev = []       # ラベルのリスト

for item in dev_data:
    x_dev_dicts.append(item['feature'])
    y_dev.append(item['label'])
    
# 特徴量のベクトル化
x_dev = vectorizer.transform(x_dev_dicts)

# 予測
y_pred = model.predict(x_dev)

# 混同行列の計算と表示
cm = confusion_matrix(y_dev, y_pred)
print(cm)