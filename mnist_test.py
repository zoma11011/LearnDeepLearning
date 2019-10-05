from keras import backend as K
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
t_train = keras.utils.to_categorical(t_train, 10)
t_test = keras.utils.to_categorical(t_test, 10)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


'''
for i in range(0,10):
 print("ラベル", x_train[i])
 plt.imshow(x_train[i].reshape(28, 28), cmap='Greys')
 plt.show()
'''