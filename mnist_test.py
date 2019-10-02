from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test,t_test) = mnist.load_data()

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

for i in range(0,10):
 print("ラベル", x_train[i])
 plt.imshow(x_train[i].reshape(28, 28), cmap='Greys')
 plt.show()
