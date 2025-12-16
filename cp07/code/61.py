import pandas as pd
import json

df_train = pd.read_csv("SST-2/train.tsv", delimiter="\t")
df_dev = pd.read_csv("SST-2/dev.tsv", delimiter="\t")

def create_data_dict(df):
    dataset_list = []
    for index, row in df.iterrows():
        text = row['sentence']
        label = row['label']
        
        feature = {}
        text_list = text.split()
        seen = set()
        for i in range(len(text_list)):
            if text_list[i] not in seen:
                seen.add(text_list[i])
                feature[text_list[i]] = 1
            else:
                feature[text_list[i]] += 1
        
        data_dict = {
            'text': text,
            'label': label,
            'feature': feature
        }
        dataset_list.append(data_dict)
    return dataset_list

train_data = create_data_dict(df_train)
dev_data = create_data_dict(df_dev)

# 出力
# ファイルに書き込む
with open("61_output_train.json", 'w') as f:
    json.dump(train_data, f, indent=4)

with open("61_output_dev.json", 'w') as f:
    json.dump(dev_data, f, indent=4)

            
        