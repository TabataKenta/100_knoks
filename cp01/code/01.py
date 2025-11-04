text="パタトクカシーー"
result=""
for i in range(len(text)):
    if i % 2 == 1:
        continue
    else:
        result+=text[i]

print(result)