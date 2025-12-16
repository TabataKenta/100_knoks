import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 学習データの処理
with open("61_output_train.json", 'r') as f:
    train_data = json.load(f)
    
x_train_dicts = [] # 特徴量の辞書のリスト
y_train = []       # ラベルのリスト

for item in train_data:
    x_train_dicts.append(item['feature'])
    y_train.append(item['label'])
    
# 学習データの特徴量のベクトル化
vectorizer = DictVectorizer()
x_train = vectorizer.fit_transform(x_train_dicts)

# 検証データの処理
with open("61_output_dev.json", 'r') as f:
    dev_data = json.load(f)

x_dev_dicts = [] # 特徴量の辞書のリスト
y_dev = []       # ラベルのリスト

for item in dev_data:
    x_dev_dicts.append(item['feature'])
    y_dev.append(item['label'])

# 検証データの特徴量のベクトル化
x_dev = vectorizer.transform(x_dev_dicts)

# 正則化パラメータ
C = [0.01, 0.1, 1, 10, 100]

# ロジスティック回帰モデルの学習
acuracies = []
for c in C:
    model = LogisticRegression(C=c, max_iter=1000) # max_iterは計算回数の上限
    model.fit(x_train, y_train)
    accuracy = model.score(x_dev, y_dev)
    acuracies.append(accuracy)

# 正則化パラメータと精度のプロット
plt.plot(C, acuracies)
plt.xscale("log") # x軸を対数スケールに設定
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Accuracy")
plt.title("Effect of C on Accuracy")
plt.show()