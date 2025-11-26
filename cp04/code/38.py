import json
import re
import spacy
import math
from collections import Counter

#既にgzファイルを解凍済み"file.json"であるとする

# クリーニング関数s
def clean_text(text):
    while True:
        original = text
        text = re.sub(r'(?s)\{\{.*?\}\}', '', text)
        if text == original:
            break
    text = re.sub(r'(?m)^\|.*', '', text)
    text = re.sub(r'}}', '', text)
    text = re.sub(r"'''''(.*?)'''''", r'\1', text)
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)
    text = re.sub(r'\[\[(?:[^|\]]*?\|)?([^|\]]+?)\]\]', r'\1', text)
    text = re.sub(r'=+(.*?)=+', r'\1', text)
    text = re.sub(r'\[https?://.*? (.*?)\]', r'\1', text)
    text = re.sub(r'(?s)<ref.*?>.*?</ref>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[(Category|カテゴリ|ファイル|File|画像|Image):.*?\]\]', '', text)
    text = re.sub(r'(?m)^\*+\s*', '', text)
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# 読み込みファイルと、書き出しファイル(2つ)を同時に開く
with open('file.json', 'r', encoding='utf-8') as f_in, \
     open('clean.txt', 'w', encoding='utf-8') as f_all, \
     open('jp_clean.txt', 'w', encoding='utf-8') as f_jp:

    for line in f_in:
        data = json.loads(line)
        title = data.get('title', '')
        text = data.get('text', '')
        
        # リダイレクトならスキップ
        if text.startswith("#REDIRECT") or text.startswith("#転送"):
            continue

        # クリーニング
        clean_article = clean_text(text)
        
        # 空でなければ処理
        if clean_article:
            # 1. 全記事ファイルへ書き込み
            f_all.write(clean_article + "\n")
            
            # 2. 日本の記事なら、専用ファイルへも書き込み
            if title == "日本":
                f_jp.write(clean_article + "\n")
                
nlp = spacy.load('ja_ginza')

# --- 1. IDFの計算（全記事 clean.txt を読む） ---
df_counter = Counter()
total_docs = 0

with open('clean.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        
        total_docs += 1
        
        # この記事に出てくる名詞のセットを作る
        words_in_doc = set()
        doc = nlp(line)
        for token in doc:
            if token.pos_ == "NOUN":
                words_in_doc.add(token.lemma_)
        
        # DFをカウント
        for word in words_in_doc:
            df_counter[word] += 1

# --- 2. TFの計算（日本の記事 jp_clean.txt を読む） ---
print("日本の記事からTFを計算中...")
tf_counter = Counter()

with open('jp_clean.txt', 'r', encoding='utf-8') as f:
     for line in f:
        line = line.strip()
        if not line:
            continue
            
        doc = nlp(line)
        for token in doc:
            if token.pos_ == "NOUN":
                tf_counter[token.lemma_] += 1

# --- 3. TF-IDFの計算と表示 ---
result = []
for word, tf in tf_counter.items():
    df = df_counter[word]
    # IDF = log(総記事数 / (その単語が出る記事数 + 1))
    idf = math.log(total_docs / (df + 1))
    tfidf = tf * idf
    
    result.append((word, tf, idf, tfidf))

# TF-IDFスコアが高い順にソート
result.sort(key=lambda x: x[3], reverse=True)

# 上位20個を表示
print(f"\n{'単語':<10} {'TF':<5} {'IDF':<8} {'TF-IDF'}")
print("-" * 40)
for item in result[:20]:
    word, tf, idf, tfidf = item
    print(f"{word:<10} {tf:<5} {idf:<8.4f} {tfidf:.4f}")
