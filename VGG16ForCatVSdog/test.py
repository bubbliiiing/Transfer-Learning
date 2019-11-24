from model.VGG16 import VGG16

model = VGG16(2)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)