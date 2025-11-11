def count_line(filename):
    cnt=0
    with open(filename,'r') as f:
        for line in f:
            cnt+=1
    return cnt

def file_N_split(N):
    total_line=count_line('popular-names.txt')
    cnt_split=total_line//N
    
    text=[]
    cnt=0
    file_num=1
    with open('popular-names.txt','r') as f:
        for line in f:
            cnt+=1
            text.append(line)
            if cnt==cnt_split:
                with open(f'15output_{file_num}.txt','w') as f2:
                    f2.write(''.join(text))
                    text=[]
                    file_num+=1
                    cnt=0
    if text:
        with open(f'15output_{file_num}.txt','w') as f2:
            f2.write(''.join(text))
            
            
