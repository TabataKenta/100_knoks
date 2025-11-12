def head_file(N):
    cnt = 0
    result=[]
    with open('popular-names.txt', 'r') as f:
        for line in f:
            cnt += 1
            if cnt <= N:
                result.append(line.split("\t")[0])
            else:
                break
    return result

print(head_file(10))
#unix: head -n 10 popular-names.txt | cut -f1
