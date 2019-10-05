import numpy as np
import matplotlib.pylab as plt
import base_funcs as bf

'''
x = np.arange(-10.0, 10.0, 0.1)

#y = bf.step_function(x)
y = bf.sigmoid(x)
print(y)

plt.plot(x, y)
plt.ylim(-0.1,1.1)
plt.show()
'''

'''
def function1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function2(x):
    return x[0] ** 2 + x[1] ** 2


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(bf.mean_squared_error(np.array(y), np.array(t)))

print(bf.numerical_gradient(function2, np.array([3.0, 4.0])))
print(bf.numerical_gradient(function2, np.array([0.0, 2.0])))
print(bf.numerical_gradient(function2, np.array([3.0, 0.0])))

init_x = np.array([-3.0, 4.0])
print(bf.gradient_descent(function2, init_x=init_x, lr=0.1, step_num=100))

'''

a = np.random.choice(10,5)
b = np.array([0,11,12,13,14,15,16,17,18,19])
print(a)
print(b[a])
