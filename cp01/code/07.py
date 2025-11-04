def generate_text(x, y, z):
    text=f'{x}時の{y}は{z}'
    return text

x=12
y="気温"
z=22.4

print(generate_text(x,y,z))