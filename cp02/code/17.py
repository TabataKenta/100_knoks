def remove_duplication(lst):
    seen=set()
    result=[]
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
    

lst=[]
with open('popular-names.txt', 'r') as f:
        for line in f:
            lst.append(line.split("\t")[0])

unique_list=remove_duplication(lst)
print(unique_list)        
