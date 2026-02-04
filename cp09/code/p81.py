from transformers import pipeline

def top_mask_filler(text):
    # "fill-mask": 「マスクされた部分を埋める」というタスクを指定
    mask_filler = pipeline("fill-mask", model="bert-base-uncased")
    results = mask_filler(text)
    return results[0]['token_str']

if __name__ == "__main__":
    text = "The movie was full of [MASK]."
    predicted_token = top_mask_filler(text)
    print(f"Predicted token: {predicted_token}")
