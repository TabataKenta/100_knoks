import json
import re

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            article = data['text']
            break

pattern1 = (
    r'(?s)'       # ドットが改行文字にもマッチするようにするフラグ(改行を無視)
    r'{{基礎情報'  # '{{基礎情報' という固定の文字列で開始
    r'(.*?)'     # グループ1: 基礎情報の内容全体。最短一致(.*?)で、次のパターンまでを取得
    r'}}'         # '}}' という固定の文字列で終了
)
#始まりの位置を特定
index = re.search(pattern1, article)
start=index.start()

#終わりの位置を特定
level=1 #入れ子の深さを数えるカウンタ(  "{{基本情報"  の時点で深さ１)
for i in range(start+2, len(article)):
    if article[i:i+2]=='{{':
        level+=1
    elif article[i:i+2]=='}}':
        level-=1
        
        if level==0:
            end=i+2
            break
    
#文字列を取得
info_box=article[start:end]
#リストに分割
lines = info_box.splitlines()
#辞書に格納
info_dict={}
pattern2=(
    r'\|' # 行の先頭にある'|'
    r'(.+?)' # グループ1: フィールド名。最短一致で取得
    r'\s*=\s*' # '='の前後の空白を許容
    r'(.+)' # グループ2: フィールド値。行末まで取得
)
last_key=None
for line in lines:
        match=re.match(pattern2, line)
        if match:
            key=match.group(1).strip()
            value=match.group(2).strip()
            info_dict[key]=value
            last_key=key
            
        elif last_key:
            info_dict[last_key]+='\n'+line.strip()

#強調マークアップの除去(長いものから処理)
emphasis_clean_dict={}
# 辞書の各要素をループで処理
for key, value in info_dict.items():
    # 1.強い強調 ('''''...''''') を除去
    cleaned_value = re.sub(r"'''''(.*?)'''''", r'\1', cleaned_value)
    
    # 2.強調('''...''') を除去
    cleaned_value = re.sub(r"'''(.*?)'''", r'\1', cleaned_value)
    
    # 3.軽い強調('...') を除去
    cleaned_value = re.sub(r"''(.*?)''", r'\1', cleaned_value)
    
    # クリーニングしたキーと値を新しい辞書に格納
    emphasis_clean_dict[key] = cleaned_value

#結果表示
for key, value in emphasis_clean_dict.items():
    print(f'{key}: {value}')
