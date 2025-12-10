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
        
        # 対象セクション内の行のみ処理を実行
        if target_section:
            words=line.split()
            
            try:
                # ベクトル計算: vec(B) - vec(A) + vec(C) に最も近い単語を探す
                # positive=[足すベクトル], negative=[引くベクトル]
                vec=model.most_similar(positive=[words[1],words[2]],negative=[words[0]],topn=1)
                similar_word=vec[0][0] # 最も類似度が高い単語（予測結果）
                score=vec[0][1]        # その類似度スコア
                
                # 次の課題で正解率を計算するため、問題(A,B,C)・正解(D)・予測結果・スコアを並べて出力
                # 順序: A B C 正解(D) 予測(Prediction) 類似度(Score)
                print(f"{words[0]} {words[1]} {words[2]} {words[3]} {similar_word} {score}")
                
            except KeyError:
            # 辞書にない単語があった場合はスキップ
              pass            
