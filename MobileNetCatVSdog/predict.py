import numpy as np
from keras import backend as K
from PIL import Image

from model.mobileNet import MobileNet

def get_classes():
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1].replace("\n","") for l in f.readlines()]
    return synset

if __name__ == "__main__":
    #---------------------------------------------#
    #   区分的类的个数
    #   猫+狗=2
    #---------------------------------------------#
    NCLASSES = 2

    model = MobileNet(classes=NCLASSES)
    #--------------------------------------------------#
    #   载入权重，训练好的权重会保存在logs文件夹里面
    #   我们需要将对应的权重载入
    #   修改model_path，将其对应我们训练好的权重即可
    #   下面只是一个示例
    #--------------------------------------------------#
    model.load_weights("./logs/middle_one.h5")

    img = Image.open("./data/image/train/cat.0.jpg")
    img = img.resize((224, 224), Image.BICUBIC)
    img = np.expand_dims(np.array(img)/255, axis = 0)

    classes = get_classes()
    print(classes[np.argmax(model.predict(img)[0])])
