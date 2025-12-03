import os
from dotenv import load_dotenv
import google.generativeai as genai
import csv
import time

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

# 4. CSV読み込み
questions_data = []
with open('global_facts.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 6:
            continue
        questions_data.append({
            "question": row[0],
            "options": f"A. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}",
            "answer": row[5]
        })

# 正解数カウント用
correct_count = 0
total_count = 0

print(f"-------採点開始（全{len(questions_data)}問）-------")

for i, item in enumerate(questions_data):
    prompt = f"""
以下は多肢選択式の問題です。正解の記号（A, B, C, D）のみを出力してください。
余計な解説は不要です。

問題: {item['question']}
{item['options']}

答え:
""" 
    # 6. 生成実行
    try:
        response = model.generate_content(prompt)
        
        # AIの答え（余白削除）
        prediction = response.text.strip()
            
        # 正解（CSVの答え）
        ground_truth = item['answer'].strip()
        
        # 判定
        if prediction:
            is_correct = prediction.startswith(ground_truth)
        else:
            is_correct = False
        
        if is_correct:
            correct_count += 1
            result_mark = "⭕️"
        else:
            result_mark = "❌"
        
        print(f"問{i+1}: {result_mark} (AI: {prediction} / 正解: {ground_truth})")
        total_count += 1
        
        # 無料枠対策
        time.sleep(4)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        time.sleep(4)

# 4. 最終結果の表示
if total_count > 0:
    accuracy = correct_count / total_count
    print(f"\n最終結果: 正解率 {accuracy:.2%} ({correct_count}/{total_count})")
