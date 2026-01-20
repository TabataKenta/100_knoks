import gensim
import numpy as np

def load_pretrained_embeddings(model):
    vocab_size = len(model.index_to_key) + 1  #語彙数 + 1 (パディング用)
    d_emb=model.vector_size #埋め込み次元数
    vocab = model.index_to_key  #語彙リスト
    
    # vocab_size x d_embの行列を作成(全要素0)
    embedding_matrix = np.zeros((vocab_size, d_emb),dtype=np.float32)
    
    # トークンIDとトークンの双方向の対応付け
    token_to_id = {'<PAD>': 0} 
    id_to_token = {0: '<PAD>'}

    for i, token in enumerate(vocab):
        embedding_matrix[i + 1] = model[token]  #インデックス1から単語ベクトルを格納
        token_to_id[token] = i + 1
        id_to_token[i + 1] = token
    return embedding_matrix, token_to_id, id_to_token

if __name__ == '__main__':
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    E, t2id, id2t = load_pretrained_embeddings(model)
    
    results = [
        f"次元数: {E.shape[1]}",
        f"語彙数: {E.shape[0]}",
        f"埋め込み行列の形状: {E.shape}",
        f"ID 100の単語: {id2t[100]}",
    ]
    
    with open('p70_result.txt', 'w') as f:
        for line in results:
            print(line)
            f.write(line + '\n')
            

