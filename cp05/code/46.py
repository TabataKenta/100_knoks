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

# 4. プロンプト
# 川柳のお題「冬」
prompt = f"""
お題「冬」
このお題で川柳の案を１０個作成してください。
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
