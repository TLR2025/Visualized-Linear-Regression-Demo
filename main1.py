import numpy as np 
import matplotlib.pyplot as plt 
from IPython.display import clear_output
import time

data = np.loadtxt("ex1data1.txt",delimiter=',')
x = data[:, 0]  
y = data[:, 1]  
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())
plt.ion()
alpha = 0.003
w = 0
b = 0
index = 0
hisJ = []
def jfwc(a,b):
    global index
    index += 1
    sum = 0.0
    for i in range(a.shape[0]):
        sum += (a[i]-b[i])*(a[i]-b[i])
    sum /= a.shape[0]
    return sum
while(True):
    error = w*x+b-y
    d_w = (error*x).mean()
    d_b = error.mean()
    w -= alpha * d_w
    b -= alpha * d_b

    clear_output(wait=True) 
    plt.figure(1)
    plt.clf()
    plt.scatter(x, y)
    plt.grid(True)
    plt.plot([x.min(), x.max()], [x.min()*w + b, x.max()*w + b], 
         color='red', label='Hypothesis: y=wx+b')
    plt.figure(2)
    plt.clf()
    hisJ.insert(index, [jfwc(w*x+b,y)]) 
    plt.plot(range(index),hisJ)
    plt.pause(1)
