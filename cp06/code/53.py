import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

A = model['Spain']
B = model['Madrid']
C = model['Athens']
vec = A - B + C

result = model.most_similar(vec, topn=10)

for word in result:
    print(word)
