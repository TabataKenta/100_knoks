def display_tail_file(N):
    cnt = 0
    with open('popular-names.txt', 'r') as f:
        lines=f.readlines()   # readlines()で全行をリストとして取得
        for line in lines[-N:]: # リストのスライス機能で後ろからN行を取得
            print(line, end='') # printだと自動的に改行する→end=''を付けて二重改行を防ぐ
display_tail_file(10)
