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

# k-meansクラスタリング
from sklearn.cluster import KMeans
    
# クラスタ数を5に設定
kmeans = KMeans(n_clusters=5, random_state=0).fit(country_vec)

# 各国とクラスタ番号を表示
for i in range(len(country)):
    print(f"{country[i]} {kmeans.labels_[i]}")
