from transformers import pipeline

def top10_mask_filler(text):
    mask_filler = pipeline("fill-mask", model="bert-base-uncased")
    # pipeline("fill-mask") は、デフォルトでは上位5件しか返さない仕様
    # そこで、top_k引数で10件取得するように指定
    results = mask_filler(text,top_k=10)
    return [(r['token_str'], r['score']) for r in results[:10]]

if __name__ == "__main__":
    text = "The movie was full of [MASK]."
    predicted_tokens = top10_mask_filler(text)
    print(f"Predicted tokens: {predicted_tokens}")
