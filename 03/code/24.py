import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            article = data['text']
            break
        
pattern1=(
    r'\[\[(ファイル|File|画像|Image):' # '[[ファイル:' または '[[File:' または '[[画像:' または '[[Image:' という固定の文字列で開始 
    r'(.*?)'         # グループ1: ファイル名本体。
    r'(?:\|.*)?'     # グループ2: '|'とそれ以降の文字列全体。'末尾の'?'で、このグループが無くてもOK
    r'\]\]'         # ']]' という固定の文字列で終了
)

pattern2=(
    r'(?s)'       # ドットが改行文字にもマッチするようにするフラグ(改行を無視)
    r'<gallery>'  # '<gallery>' という固定の文字列で開始
    r'(.*?)'      # グループ1: ギャラリー内の内容
    r'</gallery>' # '</gallery>' という固定の文字列で終了 
)

# [('ファイル', 'example.jpg'), ...]の形
results1 = re.findall(pattern1, article)
file1=[]
for result in results1:
    file1.append(result[1])

# 長い文字列が１つだけあるリスト
results2 = re.findall(pattern2, article)
file2=[]

# results2が空でないことを確認
if results2:
    # リストの先頭（長い文字列）取り出して、改行で分割したリストに変換
    s1=results2[0].splitlines()
    
    for line in s1:
        part=line.split('|',1) # '|'で分割、最大1回
        file2.append(part[0].strip()) # ファイル名部分を取り出して、前後の空白を削除してリストに追加
    
file=file1 + file2  

for filename in file:
    print(filename)
