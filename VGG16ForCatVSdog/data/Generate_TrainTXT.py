import os 


with open('./data/train.txt','w') as f:
    after_generate = os.listdir("./data/image/train")
    for image in after_generate:
        if image.split(".")[0]=='cat':
            f.write(image + ";" + "0" + "\n")
        else:
            f.write(image + ";" + "1" + "\n")
