import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from dataset.mnist import load_mnist
from PIL import Image
import numpy
import pickle


(X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

print(X_train.shape)
print(y_train.shape)

train_size = X_train.shape[0]
batch_size = 10
batch_mask = numpy.random.choice(train_size, batch_size)
X_batch = X_train[batch_mask]
y_batch = y_train[batch_mask]