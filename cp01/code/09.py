import random

text="I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
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

result=""
s=""
for i in range(len(word)):
    if len(word[i])<=4:
        result+=word[i]
        result+=" "
    else:
        s=word[i]
        result+=s[0]
        ran_s= s[1:-1]
        lst=list(ran_s)
        random.shuffle(lst)
        result+="".join(lst)
        result+=s[-1]
        result+=" "
        
print(result)       
            
        
        