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
以下の問いかけに対する応答を生成せよ。

つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
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
