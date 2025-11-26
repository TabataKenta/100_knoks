import spacy
from collections import Counter
import matplotlib.pyplot as plt # グラフ描画用ライブラリ

# 1. データの集計（問題36, 38のロジックを再利用）
# -------------------------------------------
nlp = spacy.load('ja_ginza')
words_counter = Counter()

# clean.txt を1行ずつ読み込んでカウント
with open('clean.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        
        doc = nlp(line)
        for token in doc:
            if token.pos_ == "NOUN":
                words_counter[token.lemma_] += 1

# 2. データの整理
# -------------------------------------------
# most_common() を使って頻度の高い順に (単語, 回数) のリストをとる
sorted_counts = words_counter.most_common()

# グラフに必要な「順位」と「回数」のリスト
ranks = []   # 順位 (横軸)
counts = []  # 回数 (縦軸)

for i, (word, count) in enumerate(sorted_counts, 1):
    ranks.append(i)      # 1位, 2位, 3位...
    counts.append(count) # 1000回, 500回, ...

# 3. グラフの描画 (matplotlib)
# -------------------------------------------
# 散布図を描く
plt.scatter(ranks, counts, s=10) # sは点のサイズ

# 両方の軸を対数(log)スケールにする
plt.xscale('log')
plt.yscale('log')

# ラベル付け
plt.xlabel('Rank')      # 順位
plt.ylabel('Frequency') # 出現頻度
plt.title('Zipf Law')   # タイトル

# 画像として保存
plt.savefig('ans39.png')
print("グラフを ans39.png に保存しました。")
