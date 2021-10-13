import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from dataset.mnist import load_mnist
from PIL import Image
import numpy


def img_show(img):
    pil_img = Image.fromarray(numpy.uint8(img))
    pil_img.show()

(X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=False)

img = X_train[0]
label = y_train[0]
print(label)

img = img.reshape(28, 28)
img_show(img)