from model.mobileNet import MobileNet

if __name__ == "__main__":
    model = MobileNet(classes=2)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
