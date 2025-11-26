import spacy
from collections import Counter

# 36.pyによりテキスト"clean.text"ができていることを前提とする。
       
# GiNZAのロード
nlp = spacy.load('ja_ginza')

# 単語を数えるためのカウンターを用意
words_counter = Counter()

# 保存したファイルを読み込む
with open('clean.txt', 'r', encoding='utf-8') as f:
    for line in f:
         # 改行などを除去
        line = line.strip()
        # 空行ならスキップ
        if not line:
            continue
        
        # 1行ごとに解析を実行
        doc = nlp(line)
        
        for token in doc:
            # 品詞(pos_)が「名詞(NOUN)」または「固有名詞(PROPN)」の場合のみカウント
            if token.pos_ in ["NOUN", "PROPN"]:
                words_counter[token.lemma_]+=1

# 結果の表示（頻度上位20単語）
print("--- 出現頻度ベスト20 ---")
for word, count in words_counter.most_common(20):
    print(f"{word}\t{count}")
