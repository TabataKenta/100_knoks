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
# 直前に処理したフィールド名（キー）を記憶しておくための変数を初期化
last_key = None

# 分割した行を1行ずつループ処理
for line in lines:
    # 現在の行が '|キー = 値' のパターンに一致するか試す
    match = re.match(pattern2, line)
    
    # パターンに一致した場合（= 新しいフィールドの始まり）
    if match:
        # グループ1からキーを抽出し、前後の空白を除去
        key = match.group(1).strip()
        # グループ2から値を抽出し、前後の空白を除去
        value = match.group(2).strip()
        # 辞書に新しいキーと値のペアを格納
        info_dict[key] = value
        # 次の行が値の続きだった場合に備え、現在のキーを記憶しておく
        last_key = key
            
    # パターンに一致せず、かつ直前に処理したキーが存在する場合（= 値の続きの行）
    elif last_key:
        # 記憶しておいたキーに対応する値に、改行を加えて現在の行の内容を追記する
        info_dict[last_key] += '\n' + line.strip()

print(info_dict)
