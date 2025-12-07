import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

result = model.most_similar('United_States')

for i in range(10):
    print(result[i])
