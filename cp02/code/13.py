def tab_to_space(N):
    cnt = 0
    tabsize=1
    with open('popular-names.txt', 'r') as f:
        for line in f:
            cnt += 1
            if cnt <= N:
                print(line.expandtabs(tabsize),end='')                
            else:
                break

tab_to_space(10)
