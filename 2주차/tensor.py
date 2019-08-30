from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt





for i in range(3):
    x,y = mnist.train.next_batch(1)
    # x = x + np.full((1,784),np.random.normal(0,0.1,100))
    # print(np.random.normal(0,0.1,100))
    noise = np.random.normal(0,0.1,x.shape) /2
    x = x + noise
    
    mnist_image = np.array(x).reshape((28,28))
    # plt.title("label : "+str(np.where(y[0]==1)[0][0]))
    plt.imshow(mnist_image,cmap="gray")
    plt.show()


