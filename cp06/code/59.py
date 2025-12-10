import gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('questions-words.txt', 'r') as f:
    target_section = False
    country = []
    
    for line in f:
        if line.startswith(':'):
            if 'capital-common-countries' in line:
                target_section = True
            else:
                target_section = False
            continue 
        
        if target_section:
            words = line.split()
            country.append(words[1])
            country.append(words[3])

# 重複を除去
country = list(set(country))

# 国のベクトルを取得
country_vec = []
for i in country:
    country_vec.append(model[i])

# リストをNumPy配列に変換
country_vec = np.array(country_vec)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embedded = tsne.fit_transform(country_vec)

# 2次元プロットの描画
plt.figure(figsize=(10, 10))

# 1. 点をプロット
plt.scatter(embedded[:, 0], embedded[:, 1])

# 2. 各点に国名を書き込む
for i, country_name in enumerate(country):
    plt.annotate(country_name, (embedded[i, 0], embedded[i, 1]))

# 3. 画像として保存    
plt.savefig('result_59.png')
