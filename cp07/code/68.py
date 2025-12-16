import joblib
import pandas as pd

# 62.pyで保存したモデルとベクトル化器の読み込み
model = joblib.load('62model.joblib')
vectorizer = joblib.load('62vectorizer.joblib')

# 単語リストの取得
feature_names = vectorizer.get_feature_names_out()

# 重みリストの取得
weights = model.coef_[0]

df = pd.DataFrame({'単語': feature_names, '重み': weights})

# 重みが高い順に上位10件を表示
positive = df.sort_values(by="重み", ascending=False).head(20) # 降順
print("=== ポジティブな単語トップ20 ===")
print(positive)

# 重みが低い順に上位10件を表示
negative = df.sort_values(by="重み", ascending=True).head(20) # 昇順
print("=== ネガティブな単語トップ20 ===")
print(negative)