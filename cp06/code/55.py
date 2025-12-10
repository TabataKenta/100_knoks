import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# 集計用変数の初期化
sem_correct = 0  # 意味的: 正解数
sem_total = 0    # 意味的: 合計数
syn_correct = 0  # 文法的: 正解数
syn_total = 0    # 文法的: 合計数

# format: 入力1 入力2 入力3 正解 予測 類似度スコア
with open('54.txt', 'r') as f:
    for line in f:
        cols=line.split()
        target = cols[3]      # 正解
        prediction = cols[4]  # 予測
        
        # 意味的セクションの集計
        sem_total += 1
        if target == prediction:
            sem_correct += 1

current_section = ""
# 文法セクションは54.txtではできないので、ここで再度処理
# format: Athens Greece Baghdad Iraq （首都 国 首都 国）
with open('questions-words.txt', 'r') as f:
    for line in f:
        # セクション判定
        if line.startswith(':'):
            current_section = line.strip()
            continue

        # すでに54.txtで計算済みのセクションはスキップ
        if ': capital-common-countries' in current_section:
            continue
        
        # 計算処理
        words = line.split()
        try:
            vec = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)
            prediction = vec[0][0]
            target = words[3]
            
            is_correct = (prediction == target)

            # 振り分け
            if 'gram' in current_section:
                # 文法的アナロジー
                syn_total += 1
                if is_correct:
                    syn_correct += 1
            else:
                # 意味的アナロジー
                sem_total += 1
                if is_correct:
                    sem_correct += 1

        except KeyError:
            pass        
# 正解率の表示
if sem_total > 0:
    print(f"意味的アナロジー正解率: {sem_correct / sem_total:.3f} ({sem_correct}/{sem_total})")
if syn_total > 0:
    print(f"文法的アナロジー正解率: {syn_correct / syn_total:.3f} ({syn_correct}/{syn_total})")

