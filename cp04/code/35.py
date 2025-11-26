import spacy
from spacy import displacy

nlp=spacy.load('ja_ginza') #GiNZAのロード

target="メロスは激怒した。"
doc = nlp(target)

# 描画データ(HTML)を変数に受け取る
html=displacy.render(doc, style="dep", options={"compact":True}, jupyter=False)

# graph.htmlとして保存
with open("35output.html", "w", encoding="utf-8") as f:
    f.write(html)

print("35output.htmlの保存完了")
