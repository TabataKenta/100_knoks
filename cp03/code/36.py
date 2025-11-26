import gzip
import json
import re
import spacy
from collections import Counter

source_file = "jawiki-country.json.gz"
target_file = "file.json"

# 解凍処理
with gzip.open(source_file, mode="rb") as gzip_file:
    content = gzip_file.read()
    with open(target_file, mode="wb") as decompressed_file:
        decompressed_file.write(content)

# クリーン関数
def clean_text(text):
     # 1. テンプレート（{{...}}）の再帰的除去
    # 基礎情報、langタグ、Mainタグなどを消去
    while True:
        original = text
        text = re.sub(r'(?s)\{\{.*?\}\}', '', text)
        if text == original:
            break

    # 2. 基礎情報の残りカス（行頭の|や=がある行）を除去
    text = re.sub(r'(?m)^\|.*', '', text)
    
    # 3. 閉じ括弧の残りカスを除去
    text = re.sub(r'}}', '', text)

    # 4. 強調マークアップ（'''''...'''''）の除去
    text = re.sub(r"'''''(.*?)'''''", r'\1', text)
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)
    
    # 5. 内部リンク（[[記事名|表示名]]）の除去 -> 表示名だけ残す
    # [[A|B]] -> B, [[A]] -> A
    text = re.sub(r'\[\[(?:[^|\]]*?\|)?([^|\]]+?)\]\]', r'\1', text)
    
    # 6. 見出し（== ... ==）の除去 -> 中身だけ残すか、行ごと消すか
    # ここでは文脈を繋げるため、記号だけ消して中身は残します
    text = re.sub(r'=+(.*?)=+', r'\1', text)

    # 7. 外部リンク、HTMLタグ、Category、ファイル参照の除去
    text = re.sub(r'\[https?://.*? (.*?)\]', r'\1', text) # 外部リンク
    text = re.sub(r'(?s)<ref.*?>.*?</ref>', '', text)     # 注釈
    text = re.sub(r'<[^>]+>', '', text)                   # その他のHTMLタグ
    text = re.sub(r'\[\[(Category|カテゴリ|ファイル|File|画像|Image):.*?\]\]', '', text)
    
    # 8. 箇条書き記号（*）などの整理
    text = re.sub(r'(?m)^\*+\s*', '', text) # 行頭の*を消す

    # 9. 空行の整理
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# メイン処理
all_articles = []
with open('file.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        article = data['text']
        
        # リダイレクトページならスキップ（#REDIRECT や #転送 で始まる行は無視）
        if article.startswith("#REDIRECT") or article.startswith("#転送"):
            continue
        
        clean_article=clean_text(article)
        
        if clean_article:
            all_articles.append(clean_article)



with open('clean.txt','w',encoding='utf-8') as f:
    for text in all_articles:
        f.write(text)
        
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
            words_counter[token.lemma_]+=1

# 結果の表示（頻度上位20単語）
print("--- 出現頻度ベスト20 ---")
for word, count in words_counter.most_common(20):
    print(f"{word}\t{count}")

