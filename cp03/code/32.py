import spacy

nlp=spacy.load('ja_ginza') #GiNZAのロード

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""
doc = nlp(text)

# 1番目から、最後から2番目までをループする（最初と最後は前後に単語がないため）
for i in range(1,len(doc)-1):
    if doc[i].pos_ == "ADP" and doc[i].text=="の":
        print(doc[i-1].text+"の"+doc[i+1].text)
    
