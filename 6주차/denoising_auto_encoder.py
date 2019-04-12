#2014000082
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,784])


#dropout
keep_prob = tf.placeholder(tf.float32)

#layer
W1 = tf.Variable(tf.random_uniform([784,300],-1,1))
b1 = tf.Variable(tf.random_uniform([300],-1,1))
L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1,keep_prob)


W2 = tf.Variable(tf.random_uniform([300,64],-1,1))
b2 = tf.Variable(tf.random_uniform([64],-1,1))
L2 = tf.sigmoid(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2,keep_prob)


W3 = tf.Variable(tf.random_uniform([64,300],-1,1))
b3 = tf.Variable(tf.random_uniform([300],-1,1))
L3 = tf.matmul(L2,W3)+b3
L3 = tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.random_uniform([300,784],-1,1))
b4 = tf.Variable(tf.random_uniform([784],-1,1))
model = tf.matmul(L3,W4)+b4
model = tf.sigmoid(model)

# minimize the squared error
cost = tf.reduce_mean(tf.pow(Y - model, 2))

#adam optimizer
optimizer = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    use_locking=False,
    name='Adam'
).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 30
total_batch = int(mnist.train.num_examples/batch_size)


def normalize(M):
    max = np.amax(M)
    min = np.amin(M)
    for i in range(M.shape[1]):
        M[0][i] = (M[0][i] - min)/(max - min)

for epoch in range(10):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        noise = np.random.normal(0,0.1,batch_xs.shape)
        batch_xs_noisy = batch_xs + noise
        feed_dict = {X:batch_xs_noisy,Y:batch_xs,keep_prob:0.7}
        _,cost_val = sess.run([optimizer,cost],feed_dict=feed_dict)
        total_cost += cost_val

    print("Epoch :",'%04d' % (epoch + 1),
        'Avg. cost=', '{:.3f}'.format(total_cost/total_batch))

for i in range(3):
    xs,_ = mnist.test.next_batch(1)    
    noise = np.random.normal(0,0.1,xs.shape)
    xs_noise = xs + noise    
    result = sess.run(model,feed_dict={X:xs_noise,keep_prob:1})
    image = np.array(xs).reshape((28,28))
    image_noise = np.array(xs_noise).reshape((28,28))
    image_result = np.array(result).reshape((28,28))

    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.imshow(image,cmap="gray")
    fig.add_subplot(1,3,2)
    plt.imshow(image_noise,cmap="gray")
    fig.add_subplot(1,3,3)
    plt.imshow(image_result,cmap="gray")
    plt.show()



