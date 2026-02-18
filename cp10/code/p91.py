"""
91. 続きのテキストの予測
“The movie was full of”に続くテキストを複数予測せよ。
このとき、デコーディングの方法や温度パラメータ（temperature）を変えながら、
予測される複数のテキストの変化を観察せよ。
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_method_text(model,tokenizer,text,method='greedy',**kwargs):
    """
    GPT-2モデルを使ってテキストを生成する関数
    input:
        model: 事前学習済みの言語モデル
        tokenizer: モデルに対応するトークナイザー
        text(str): 入力テキスト
        method(str): デコーディングの方法（'greedy', 'beam', 'sample'）
        **kwargs: デコーディングの方法に応じた追加の引数
    output:
        result：生成されたテキストのリスト
     """
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # テキストをトークン化してテンソルに変換
    inputs = tokenizer(text, return_tensors='pt').to(device)
    
    # generateに渡す共通の設定
    # input_ids: 入力のトークンIDのテンソル
    # attention_mask: 注意マスクのテンソル（入力のトークン数に合わせて1のテンソルを作成する）
    common_kwargs = {
        **inputs, # **inputs で input_ids と attention_mask を両方渡す
        'max_length': 30, # 生成するテキストの最大長
        # パディングトークンID（max_lengthに達する前に、生成が終了した場合に、残りの部分を埋めるためのトークン）を指定。
        # GPT-2はパディングトークンを持たないため、EOSトークン（「文の終わり」を意味する特別なトークン）を使用
        'pad_token_id': tokenizer.eos_token_id, 
        **kwargs # デコーディングの方法に応じた追加の引数
    }
    
    #================================
    # methodに応じたデコーディングの設定
    #================================
    
    # greedy: 確率の高いトークンを選択する
    # - do_sample: Falseに設定することで、確率の高いトークンを選択する。
    # - num_beams=1: ビームサーチのビーム数を1に設定することで、グリーディーサーチと同様の動作をする。
    if method == 'greedy':
        common_kwargs['do_sample'] = False
        common_kwargs['num_beams'] = kwargs.get('num_beams', 1) # num_beamsが指定されていない場合は、デフォルトで1に設定する
        
    # beam: ビームサーチを使用して複数の候補を生成する
    # - num_beams: ビームサーチのビーム数
    # - early_stopping: 早期停止を有効にするかどうか
    elif method == 'beam':
        common_kwargs['do_sample'] = False
        common_kwargs['num_beams'] = kwargs.get('num_beams', 5)   # num_beamsが指定されていない場合は、デフォルトで5に設定する
        common_kwargs['early_stopping'] = True # 早期停止を有効
    
    # sample: 確率に基づいてランダムにトークンを選択する
    # - do_sample: Trueに設定することで、確率に基づいてランダムにトークンを選択する。
    elif method == 'sample':
        common_kwargs['do_sample'] = True
    
    # モデルを評価モードに設定
    model.eval()
    with torch.no_grad():
        # model.generate()は、入力されたトークンIDをモデルに入れて、テキストを生成する。
        # output_ids: 生成されたテキストのトークンIDのテンソルで、形状は(生成するテキストの数, 生成されたトークン数)。
        output_ids = model.generate(**common_kwargs)
        
        result = []
        for output in output_ids:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            result.append(generated_text)
    return result
    
if __name__ == "__main__":
    # モデルとトークナイザーの準備
    print("モデルとトークナイザーの準備中...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "The movie was full of"
    
    print("=== Greedy Decoding ===")
    greedy_result = generate_method_text(model,tokenizer,text, method='greedy')
    print(greedy_result[0])
    
    print("\n=== Beam Search Decoding(num_beams=5) ===")
    beam_result = generate_method_text(model,tokenizer,text, method='beam', num_beams=5, early_stopping=True)
    print(beam_result[0])
    
    print("\n=== Sampling Decoding (Temperature=0.1) ===")
    sample_result_temp0_1 = generate_method_text(model, tokenizer, text, method='sample', temperature=0.1)
    print(sample_result_temp0_1[0])
    
    
    print("\n=== Sampling Decoding (Temperature=0.7) ===")
    sample_result_temp0_7 = generate_method_text(model, tokenizer, text, method='sample', temperature=0.7)
    print(sample_result_temp0_7[0])
        
    print("\n=== Sampling Decoding (Temperature=1.5) ===")
    sample_result_temp1_5 = generate_method_text(model, tokenizer, text, method='sample', temperature=1.5)
    print(sample_result_temp1_5[0]) 
        
    
