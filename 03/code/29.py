import json
import re
import requests

# --- 課題25までのコード (info_dictの作成) ---

with open('jawiki-country.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            article = data['text']
            break

pattern1 = (
    r'(?s)'
    r'{{基礎情報'
    r'(.*?)'
    r'}}'
)
index = re.search(pattern1, article)
start = index.start()

level = 1
for i in range(start + 2, len(article)):
    if article[i:i+2] == '{{':
        level += 1
    elif article[i:i+2] == '}}':
        level -= 1
        if level == 0:
            end = i + 2
            break

info_box = article[start:end]
lines = info_box.splitlines()
info_dict = {}
pattern2 = (
    r'\|'
    r'(.+?)'
    r'\s*=\s*'
    r'(.+)'
)
last_key = None
for line in lines:
    match = re.match(pattern2, line)
    if match:
        key = match.group(1).strip()
        value = match.group(2).strip()
        info_dict[key] = value
        last_key = key
    elif last_key:
        info_dict[last_key] += '\n' + line.strip()

# --- 課題29 ---

# 1. 辞書から国旗画像のファイル名を取得
flag_filename = info_dict['国旗画像']

# 2. MediaWiki APIへのリクエストを準備
S = requests.Session()
URL = "https://ja.wikipedia.org/w/api.php"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{flag_filename}",
    "iiprop": "url"
}

# 3. APIにGETリクエストを送信
try:
    R = S.get(url=URL, params=PARAMS, headers=headers)
    
    # ステータスコードをチェックして、リクエストが成功したか確認
    R.raise_for_status()
    
    DATA = R.json()

    # 4. JSONレスポンスからURLを抽出
    pages = DATA["query"]["pages"]
    page_id = list(pages.keys())[0]
    
    # "imageinfo"キーが存在するかチェック
    if "imageinfo" in pages[page_id]:
        image_url = pages[page_id]["imageinfo"][0]["url"]
        # 5. 結果を出力
        print(f"国旗画像のファイル名: {flag_filename}")
        print(f"取得したURL: {image_url}")
    else:
        print(f"ファイル '{flag_filename}' の画像情報が見つかりませんでした。")


except requests.exceptions.RequestException as e:
    print(f"APIリクエスト中にエラーが発生しました: {e}")
except json.JSONDecodeError:
    print("JSONのデコードに失敗しました。APIから有効なJSONが返されませんでした。")
    print("---生のレスポンス---")
    print(R.text)
except (KeyError, IndexError):
    print("APIレスポンスの解析に失敗しました。予期しない形式の可能性があります。")
