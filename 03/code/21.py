import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            article = data['text']
            break

# [[Category:カテゴリ名|ソートキー]] の形式から「カテゴリ名」を抽出する正規表現
pattern = (
    r'\[\[Category:'  # '[[Category:' という固定の文字列で開始
    r'(.*?)'         # グループ1: カテゴリ名本体。最短一致(.*?)で、'|'の手前までを取得
    r'(\|.*)?'       # グループ2: '|'とそれ以降の文字列全体。末尾の'?'で、このグループが無くてもOK
    r'\]\]'         # ']]' という固定の文字列で終了
)         

category_lines=[]

# 記事本文を1行ずつのリストに分割する
for line in article.split('\n'):
    # re.search() を使って、その行にパターンが存在するかどうかをチェック
    if re.search(pattern,line):
         # パターンが見つかったら、その行全体をリストに追加
        category_lines.append(line)

# 結果を出力
for line in category_lines:
    print(line)
