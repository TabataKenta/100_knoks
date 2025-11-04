def cipher(text):
    result=""
    for i in range(len(text)):
        if text[i].islower():
            result+=chr(219-ord(text[i]))
        else:
            result+=text[i]
    return result

words="Hello world!"

encord=cipher(words)
print("暗号化：",encord) 

decord=cipher(encord)
print("複合化：",decord)
