import gensim
import pandas as pd
from scipy.stats import spearmanr

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# 'combined.csv' ファイルを読み込む
# フォーマット：Word 1,Word 2,Human (mean)
df = pd.read_csv('combined.csv')

# 単語ベクトルによる類似度の計算
vec_score=[]
for index, row in df.iterrows():
    vec=model.similarity(row['Word 1'],row['Word 2'])
    vec_score.append(vec)

# 人間の類似度    
human_score=df['Human (mean)']

# スピアマン相関係数の計算
correlation, pvalue = spearmanr(vec_score, human_score)
print(correlation)
