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

for sent in doc.sents: 
    for token in sent:
        # 出力：[係り元(子)] [TAB] [係り先(親)] 
        print(token.text + "\t" + (token.head).text)
