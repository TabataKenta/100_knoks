text="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
word=[]
tmp=""

#単語に分割してリストに格納
for i in range(len(text)):
    if text[i]==" ":
        word.append(tmp)
        tmp=""
    else:
        tmp+=text[i]

if tmp!="":
    word.append(tmp)

#辞書に格納
index=[0,4,5,6,7,8,14,15,18]
word_dict={}
s=""
for j in range(len(word)):
    if j==index:
        s=word[j]
        head=s[0]
        word_dict[head]=j+1
        s=""
    else:
        s=word[j]
        heads=s[0]+s[1]
        word_dict[heads]=j+1
        s=""

print(word_dict)