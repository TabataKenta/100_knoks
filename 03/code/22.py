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

results = re.findall(pattern, article)
# resultsはタプルのリスト[(カテゴリ名, ソートキー), ...]の形になるので、カテゴリ名（最初の要素）だけを抽出
category_names = []
for result in results:
    category_names.append(result[0])
print(category_names)
