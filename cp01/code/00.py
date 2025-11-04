text1="パトカー"
text2="タクシー"
text=""
for i in range(len(min(text1,text2))):
    text+=text1[i]+text2[i]

print(text)