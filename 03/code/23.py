import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            article = data['text']
            break

pattern=(
    r'(={2,})' # 2個以上の=をキャプチャ
    r'(.*?)'   # セクション名をキャプチャ
    r'\1'     # 最初のキャプチャと同じ数の=で終了
)

# sectionsはタプルのリスト[(=の数, セクション名), ...]の形になる
sections = re.findall(pattern, article)

for section in sections:
    sections_name=section[1].strip()  # セクション名の前後の空白を削除
    level=len(section[0]) -1          # =の数-1がセクションレベル
    print(level, sections_name)
