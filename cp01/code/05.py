def n_gram(text, n):
    s = []
    for i in range(len(text) - n + 1):
        s.append(text[i:i+n])
    return s

word="I am an NLPer"

#文字tri-gramを得る
tri=n_gram(word,3)

#単語bi-gramを得る
words=word.split()
bi=n_gram(words,2)

print("文字tri-gram:" , tri)
print("単語bi-gram:" , bi)