import random

def shuffle_lines(filename):
    with open(filename, 'r') as f:
        lines=f.readlines()#行ごとに分割したリストを取得
        
    random.shuffle(lines)
    
    with open("16output-"+filename, 'w') as f2:
        f2.writelines(lines)
        
shuffle_lines('popular-names.txt')
            
