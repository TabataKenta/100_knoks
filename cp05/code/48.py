import os
import time
import statistics
import re  # ここでインポートするのが一般的です
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("APIキーが見つかりません。.envを確認してください。")
    exit()

genai.configure(api_key=api_key)

# モデル設定（temperatureを追加して、あえて評価をばらつかせやすくする）
model = genai.GenerativeModel("models/gemini-2.5-flash")

senryu_list = [
    "コタツから 出られぬ体 鍋が呼ぶ",
    "雪かきで 腰をさすって また明日",
    "朝布団 出るのをためらい 二度寝かな",
    "日の短さ イルミネーション 街を染め",
    "カサカサの 肌に潤い また一日",
    "あったかい ココアでホッと 一人きり",
    "マフラーに 顔を埋めれば 駅はまだ",
    "初詣 賑わう道に 願を掛け",
    "コタツ出て みかんを剥けば 止まらない",
    "降り積もる 雪の静けさ 眠くなる"
]

results = {}

print(f"--- 実験開始: 全{len(senryu_list)}句 ---")

# 外側のループ
for i, senryu in enumerate(senryu_list):
    print(f"[{i+1}/{len(senryu_list)}] 評価中: {senryu}")
    scores = []
    
    # 内側のループ
    for t in range(5):
        prompt = f"""
        以下の川柳を10点満点で評価してください。
        回答は「数値のみ」を出力してください（解説不要、数字以外書くな）。
        
        川柳: {senryu}
        """
        
        try:
             response = model.generate_content(prompt)
             
             text = response.text.strip()
             
             match = re.search(r'\d+', text)
             if match:
                score = int(match.group())
                scores.append(score)
             else:
                print(f"  -> 数値読み取り失敗: {text}")
            
        except Exception as e:
            print(f"  -> エラー: {e}")
        
        # 待機
        time.sleep(3)
        
    # --- ここからインデントを戻す（内側のループの外に出す） ---
    
    # 5回の採点が終わってから結果を保存
    results[senryu] = scores
    
    # 集計と表示
    if len(scores) > 1:
        avg = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        print(f"  -> 完了: {scores} (平均: {avg:.1f}, 標準偏差: {stdev:.2f})")
    else:
        print(f"  -> 完了: {scores} (データ不足)")
    
    print("-" * 30)

print("\n=== 最終結果レポート ===")
print(f"{'川柳 (先頭10文字)':<15} | {'平均':<5} | {'標準偏差(ブレ)':<10}")
print("-" * 40)

for senryu, scores in results.items():
    if len(scores) > 1:
        avg = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        print(f"{senryu[:10]}... | {avg:.1f}  | {stdev:.2f}")
    else:
        print(f"{senryu[:10]}... | 計測失敗")
