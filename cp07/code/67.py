import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 62.pyで保存したモデルとベクトル化器の読み込み
model = joblib.load('62model.joblib')
vectorizer = joblib.load('62vectorizer.joblib')

# スコア計算用関数の定義
def calculate_score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"正解率: {accuracy:.4f}")
    print(f"適合率: {precision:.4f}")
    print(f"再現率: {recall:.4f}")
    print(f"F1スコア: {f1:.4f}")

# 学習データの処理
with open("61_output_train.json", 'r') as f:
    train_data = json.load(f)
    
x_train_dicts = [] # 特徴量の辞書のリスト
y_train = []       # ラベルのリスト

for item in train_data:
    x_train_dicts.append(item['feature'])
    y_train.append(item['label'])
    
# 特徴量のベクトル化
x_train = vectorizer.transform(x_train_dicts)

# 予測
y_pred = model.predict(x_train)

# 学習データに対するスコアの計算と表示
print("=== 学習データのスコア ===")
calculate_score(y_train, y_pred)

# 検証データの処理
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

# 検証データに対するスコアの計算と表示
print("=== 検証データのスコア ===")
calculate_score(y_dev, y_pred)