text="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
word=[]
tmp=""

for i in range(len(text)):
    if text[i]==" ":
        word.append(tmp)
        tmp=""
    else:
        tmp+=text[i]

if tmp!="":
    word.append(tmp)
    
print(word)