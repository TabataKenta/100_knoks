import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('questions-words.txt', 'r') as f:
    
    target_section = False
    
    for line in f:
        if line.startswith(':'):
            if 'capital-common-countries' in line:
                target_section = True
            else:
                target_section = False
            continue 
        
        if target_section:
            words=line.split()
            
            try:
                vec=model.most_similar(positive=[words[1],words[2]],negative=[words[0]],topn=1)
                similar_word=vec[0][0]
                score=vec[0][1]
                print(f"{words[0]} {words[1]} {words[2]} {words[3]} {similar_word} {score}")
            except KeyError:
            # 辞書にない単語があった場合はスキップ
              pass            
