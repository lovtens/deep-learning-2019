#2014000082
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

#layer
W1 = tf.Variable(tf.random_uniform([784,1000],-1,1))
b1 = tf.Variable(tf.random_uniform([1000],-1,1))
L1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_uniform([1000,700],-1,1))
b2 = tf.Variable(tf.random_uniform([700],-1,1))
L2 = tf.sigmoid(tf.matmul(L1,W2)+b2)

W3 = tf.Variable(tf.random_uniform([700,700],-1,1))
b3 = tf.Variable(tf.random_uniform([700],-1,1))
L3 = tf.sigmoid(tf.matmul(L2,W3)+b3)

W4 = tf.Variable(tf.random_uniform([700,10],-1,1))
b4 = tf.Variable(tf.random_uniform([10],-1,1))
model = tf.matmul(L3,W4)+b4
# output_softmax = tf.nn.softmax(model)
# output_argmax = tf.argmax(model,1)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=model))

optimizer = tf.train.GradientDescentOptimizer(0.015).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 25
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _,cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys})
        total_cost += cost_val

    print("Epoch :",'%04d' % (epoch + 1),
        'Avg. cost=', '{:.3f}'.format(total_cost/total_batch))

is_correct = tf.equal(tf.math.argmax(model,1),tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print("정확도",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))






