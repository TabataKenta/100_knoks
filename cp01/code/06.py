def n_gram(text, n):
    s = []
    for i in range(len(text) - n + 1):
        s.append(text[i:i+n])
    return s


x="paraparaparadise"
y="paragraph"

X=n_gram(x,2)
Y=n_gram(y,2)

X_set = set(X)
Y_set = set(Y)

X_plus_Y = X_set | Y_set    # 和集合
X_prod_Y = X_set & Y_set    # 積集合
X_diff_Y = X_set - Y_set    # 差集合

print("和集合：",X_plus_Y)
print("積集合：",X_prod_Y)
print("差集合：",X_diff_Y)

#XおよびYに’se’というbi-gramが含まれるかどうか調べる
if "se" in X:
    print("Xにseというbi-gramが含まれている")
if "se" in Y:
    print("Yにseというbi-gramが含まれている")
