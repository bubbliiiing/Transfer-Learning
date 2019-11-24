import numpy as np
import utils
import cv2
from keras import backend as K
from model.VGG16 import VGG16

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = VGG16(2)
    model.load_weights("./logs/middle_one.h5")
    img = cv2.imread("./data/image/train/cat.1.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    img = np.expand_dims(img,axis = 0)
    img = utils.resize_image(img,(224,224))
    print(utils.print_answer(np.argmax(model.predict(img))))