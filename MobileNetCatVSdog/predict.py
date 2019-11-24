import numpy as np
import utils
import cv2
from keras import backend as K
from model.mobileNet import MobileNet

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = MobileNet(classes =  2)
    model.load_weights("./logs/middle_one.h5")
    img = cv2.imread("./data/train/cat.0.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    img = np.expand_dims(img,axis = 0)
    img = utils.resize_image(img,(224,224))
    print(utils.print_answer(np.argmax(model.predict(img))))