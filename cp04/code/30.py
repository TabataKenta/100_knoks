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



for sent in doc.sents: # 文境界解析
    for token in sent: # 形態素解析
        if token.pos_ == "VERB": # 動詞のみ表示
            print(token)
