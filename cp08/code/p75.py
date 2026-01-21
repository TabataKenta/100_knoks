from torch.nn.utils.rnn import pad_sequence
import torch

def collate(batch):
    """
    batch: 辞書のリスト（例: [{'text':..., 'label':..., 'input_ids':...}, ...]）
    """
    # 1. 事例を「input_ids」が長い順に並び替える
    #  sorted 関数 を使い、key に input_ids の長さを指定
    batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)
    
    # 2. input_ids のリストを作成し、パディングする
    input_ids_list = [item['input_ids'] for item in batch]
    # batch_first=True にすると (バッチサイズ, 最大長) の形になる
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    
    # 3. label のリストを作成し、1つのテンソルにまとめる
    labels = torch.stack([item['label'] for item in batch])
    
    result = {
        'input_ids': input_ids_padded,
        'label': labels
    }
    
    return result

if __name__ == '__main__':
    # 問題文にある4つの事例を模したデータ
    test_batch = [
        {'text': 'hide new secretions from the parental units',
         'label': torch.tensor([0.]),
         'input_ids': torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594])},
        {'text': 'contains no wit , only labored gags',
         'label': torch.tensor([0.]),
         'input_ids': torch.tensor([3475, 87, 15888, 90, 27695, 42637])},
        {'text': 'that loves its characters and communicates something rather beautiful about human nature',
         'label': torch.tensor([1.]),
         'input_ids': torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964])},
        {'text': 'remains utterly satisfied to remain the same throughout',
         'label': torch.tensor([0.]),
         'input_ids': torch.tensor([987, 14528, 4941, 873, 12, 208, 898])}
    ]

    result = collate(test_batch)
    
    print("input_ids shape:", result['input_ids'].shape) # (4, 11) になるはず
    print("input_ids:\n", result['input_ids'])
    print("label:\n", result['label'])
