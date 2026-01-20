import pandas as pd
import torch
import pickle


def make_dataset(file_path, token_to_id):
    df = pd.read_csv(file_path, delimiter="\t")
    
    dataset = []
    
    for _, row in df.iterrows():
        text = row['sentence']
        label = row['label']
        
        input_ids = []
        token = text.split()
        for i in token:
            if i in token_to_id:
                input_ids.append(token_to_id[i])
        
        if len(input_ids) == 0:
            continue
        else:
            data = {
                "text": text,
                "label": torch.tensor([float(label)], dtype=torch.float32),
                "input_ids": torch.tensor(input_ids, dtype=torch.long)
            }
            dataset.append(data)

    return dataset


if __name__ == '__main__':
    with open('token_to_id.pkl', 'rb') as f:
        t2id = pickle.load(f)

    train_dataset = make_dataset('SST-2/train.tsv', t2id)
    dev_dataset = make_dataset('SST-2/dev.tsv', t2id)

    results =[
        f"訓練データ数: {len(train_dataset)}",
        f"検証データ数: {len(dev_dataset)}",
        "",
        "最初の訓練データ例:",
        f"テキスト: {train_dataset[0]['text']}",
        f"ラベル: {train_dataset[0]['label'].item()}",
        f"入力ID: {train_dataset[0]['input_ids'].tolist()}"
    ]
    
    # 保存処理
    with open('p71_result.txt', 'w') as f:
        for line in results:
            print(line)
            f.write(line + '\n')
            
    torch.save(train_dataset, 'train_dataset.pt')
    torch.save(dev_dataset, 'dev_dataset.pt')
