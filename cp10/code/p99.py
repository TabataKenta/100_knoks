"""
99. 選好チューニング
問題96のプロンプトに対して、正解の感情ラベルを含むテキストを望ましい応答、
間違った感情ラベルを含むテキストを望ましくない応答として、
事前学習済み言語モデルを選好チューニング (preference tuning) を実施せよ。
選好チューニングのアルゴリズムとしては、
近傍方策最適化 (PPO: Proximal Policy Optimization) や
直接選好最適化 (DPO: Direct Preference Optimization) などが考えられる。
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


# ==============================
# プロンプト生成関数
# ==============================
def build_prompt(sentence):
    """
    入力文から感情分析用プロンプトを作成する。
    問題96と同形式を想定。
    """
    return f"Review: {sentence}\nSentiment:"


def main():

    # ==============================
    # 1. モデルとトークナイザの準備
    # ==============================

    # 事前学習済みモデル（CausalLM）を使用
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 学習対象モデル（更新される）
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 参照モデル（固定。DPOの基準分布）
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)


    # ==============================
    # 2. SST-2 データセット読み込み
    # ==============================

    data_files = {
        "train": "SST-2/train.tsv",
        "validation": "SST-2/dev.tsv",
    }

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t"
    )


    # ==============================
    # 3. DPO用データ形式へ変換
    # ==============================

    def preprocess(example):
        """
        各データを
        - prompt
        - chosen（正解ラベル）
        - rejected（誤ラベル）
        の形式へ変換する。
        """

        prompt = build_prompt(example["sentence"])

        # 正解ラベルに応じて chosen / rejected を決定
        if example["label"] == 1:
            chosen = " positive"
            rejected = " negative"
        else:
            chosen = " negative"
            rejected = " positive"

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    train_dataset = dataset["train"].map(preprocess)


    # ==============================
    # 4. DPOの学習設定
    # ==============================

    training_args = DPOConfig(
        output_dir="./dpo-sst2",             # 保存先
        per_device_train_batch_size=8,      # バッチサイズ
        learning_rate=5e-6,                 # 学習率
        num_train_epochs=1,                 # エポック数
        beta=0.1                            # DPOの温度パラメータ
    )


    # ==============================
    # 5. DPOTrainerの作成
    # ==============================

    trainer = DPOTrainer(
        model=model,                # 学習対象モデル
        ref_model=ref_model,        # 参照モデル（固定）
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )


    # ==============================
    # 6. 学習実行
    # ==============================

    trainer.train()


    # ==============================
    # 7. モデル保存
    # ==============================

    trainer.save_model("./dpo-sst2-final")

if __name__ == "__main__":
    main()
