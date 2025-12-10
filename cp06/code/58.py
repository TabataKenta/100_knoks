import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('questions-words.txt', 'r') as f:
    
    target_section = False
    country=[]
    
    for line in f:
        if line.startswith(':'):
            if 'capital-common-countries' in line:
                target_section = True
            else:
                target_section = False
            continue 
        
        # format: Athens Greece Baghdad Iraq （首都 国 首都 国）
        if target_section:
            words=line.split()
            country.append(words[1])
            country.append(words[3])

# 重複を除去
country=list(set(country))

# 国のベクトルを取得
country_vec=[]
for i in country:
    country_vec.append(model[i])

# woard法による階層的クラスタリング
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

clustered = linkage(country_vec, method='ward')

# デンドログラムの描画
plt.figure(figsize=(15, 5))

# デンドログラムの表示
dendrogram(
    clustered,      # linkageの計算結果
    labels=country, # 国名のリスト
    leaf_rotation=90, # テキストを縦書きにして読みやすくするオプション
    leaf_font_size=8  # フォントサイズ
)

plt.savefig('result_58.png')

