def count_line(filename):
    cnt=0
    with open(filename,'r') as f:
        for line in f:
            cnt+=1
    return cnt
        
print(count_line('popular-names.txt'))

# UNIX: wc -l < popular-names.txt
