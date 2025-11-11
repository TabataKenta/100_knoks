#名前とその出現頻度をnameという辞書に格納
name={}
seen=set()
with open('popular-names.txt', 'r') as f:
        for line in f:
            name_line=line.split("\t")[0]
            if name_line not in seen:
                seen.add(name_line)
                name[name_line]=1
            else:
                name[name_line]+=1

#出現頻度を降順でソート
sort_name=sorted(name.items(),key=lambda x:x[1],reverse=True) 
print(sort_name)    
