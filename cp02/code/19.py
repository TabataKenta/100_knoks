from operator import itemgetter

lst=[]
with open('popular-names.txt', 'r') as f:
    for line in f:
        lst.append(line.split())
        
sorted_lst=sorted(lst,key=itemgetter(3),reverse=True)
print(sorted_lst)
    
