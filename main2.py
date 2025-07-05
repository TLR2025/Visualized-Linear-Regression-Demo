import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_regression
import random
import time

def makeData():
    dtX, dtY, dtCoe = make_regression(n_samples=100, n_features=2, random_state=None, noise=50.0, coef=True)
    dtB = random.uniform(-500, 500)
    dtY += dtB
    print(dtCoe, dtB)
    np.savetxt('rdm_data1', np.hstack((dtX, dtY.reshape(-1, 1))), delimiter=',', comments='') 

makeData()

data = np.loadtxt('rdm_data1',delimiter=',')
data[:,0] = data[:,0]

alpha = 0.03
theta = np.zeros(dtype=float, shape=data.shape[1])
X,Y = np.meshgrid(np.linspace(-3,3,20),np.linspace(-3,3,20))
hisJ = []

def J():
    tmp = theta[0]*data[:,0] + theta[1]*data[:,1] + theta[2] - data[:, 2]
    tmp = tmp * tmp
    return tmp.mean()

fig = plt.figure()
figJ = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axJ = figJ.add_subplot()
plt.ion()

rst = 0

for steps in range(1000):
    error = np.zeros(dtype=float, shape=data.shape[0])
    error = data[:,:-1] @ theta[:-1] + theta[-1] - data[:,-1]
    #print(error)
    theta[:-1] -= alpha * ((error @ data[:,:-1])/data.shape[0]) # theta:n error:m data:m*n
    theta[-1] -= alpha * error.mean()
    #print(theta)

    if steps%10 != 0:
        continue

    Z = X*theta[0] + Y*theta[1] + theta[2]
    ax.clear()
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.plot_surface(X, Y, Z, color='green', alpha=0.5, label="surface z ="+format(theta[0],".3f")+"x+"+format(theta[1],".3f")+"y+"+format(theta[2],".3f"))
    ax.legend()

    axJ.clear()
    tJ = J()
    hisJ.insert(steps, tJ)
    axJ.plot(range(hisJ.__len__()),hisJ)
    plt.pause(0.01)

plt.ioff()
plt.show()
