import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. 環境設定の読み込み
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("APIキーが見つかりません。.envを確認してください。")
    exit()

# 2. APIキーを設定
genai.configure(api_key=api_key)

# 3. モデルの準備
model = genai.GenerativeModel("models/gemini-2.5-flash")

# 4. プロンプト（命令）を作成
prompt = f"""
以下の問題の解答を作成せよ。ただし、解答生成はzero-shot推論とせよ。

# 問題
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

print(f"質問: {prompt}\n")
print("-------回答生成中-------\n")

# 5. 生成実行
try:
    response = model.generate_content(prompt)
    
    # 6. 結果の表示
    print(response.text)

except Exception as e:
    print(f"エラーが発生しました: {e}")
