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

# 4. 川柳の準備（問題46の出力のリスト）
senryu_list = {"コタツから 出られぬ体 鍋が呼ぶ",
"雪かきで 腰をさすって また明日",
"朝布団 出るのをためらい 二度寝かな",
"日の短さ イルミネーション 街を染め",
"カサカサの 肌に潤い また一日",
"あったかい ココアでホッと 一人きり",
"マフラーに 顔を埋めれば 駅はまだ",
"初詣 賑わう道に 願を掛け",
"コタツ出て みかんを剥けば 止まらない",
"降り積もる 雪の静けさ 眠くなる"}

# 5. プロンプト（命令）を作成
prompt = f"""
以下の川柳の面白さを10段階で評価せよ。
{senryu_list}
"""
print(f"質問: {prompt}\n")
print("-------回答生成中-------\n")
# 6. 生成実行
try:
    response = model.generate_content(prompt)
    
# 7. 結果の表示
    print(response.text)
except Exception as e:
    print(f"エラーが発生しました: {e}")
