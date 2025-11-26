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

# doc.sents で文章を「文」単位に区切り、ループ
for sent in doc.sents:
    # 各文の中にある「単語（トークン）」を1つずつチェック 
    for token in sent:
        
        # 条件判定： 「メロス」 かつ 「主語(nsubj)」 か？
        # token.text: 単語の表記そのもの
        # token.dep_: 依存関係ラベル（nsubj は nominal subject = 主語）
        if token.text=="メロス" and token.dep_=="nsubj":
            
            # token.head: その単語の係り先（親）。主語の親は「述語」
            predicate=token.head
            
            # predicate.lemma_: 述語の原形（辞書形）を表示
            print(f"{predicate.lemma_}")        
