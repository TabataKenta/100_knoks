import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

with open("61_output_train.json", 'r') as f:
    train_data = json.load(f)
    
x_train_dicts = [] # 特徴量の辞書のリスト
y_train = []       # ラベルのリスト

for item in train_data:
    x_train_dicts.append(item['feature'])
    y_train.append(item['label'])
    
# DictVectorizerを使って特徴量のベクトル化
vectorizer = DictVectorizer()
x_train = vectorizer.fit_transform(x_train_dicts)

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=1000) # max_iterは計算回数の上限
model.fit(x_train, y_train)

# モデルとベクトル化器の保存
joblib.dump(model, '62model.joblib')
joblib.dump(vectorizer, '62vectorizer.joblib')