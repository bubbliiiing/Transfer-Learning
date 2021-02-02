from model.VGG16 import VGG16

if __name__ == "__main__":
    model = VGG16(2)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)