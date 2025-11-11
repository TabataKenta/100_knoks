def display_head_file(N):
    cnt = 0
    with open('popular-names.txt', 'r') as f:
        for line in f:
            cnt += 1
            if cnt <= N:
                print(line, end='') # printだと自動的に改行する→end=''を付けて二重改行を防ぐ
            else:
                break

display_head_file(10)
