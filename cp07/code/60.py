import pandas as pd

df_train = pd.read_csv("SST-2/train.tsv", delimiter="\t")
df_dev = pd.read_csv("SST-2/dev.tsv", delimiter="\t")

train_positive = df_train[df_train['label']==1]
train_negative = df_train[df_train['label']==0]
dev_positive = df_dev[df_dev['label']==1]
dev_negative = df_dev[df_dev['label']==0]

print(f"Train Positive: {len(train_positive)}")
print(f"Train Negative: {len(train_negative)}")
print(f"Dev Positive: {len(dev_positive)}")
print(f"Dev Negative: {len(dev_negative)}")
