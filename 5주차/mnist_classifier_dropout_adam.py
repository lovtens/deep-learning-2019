#2014000082
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

#dropout
keep_prob = tf.placeholder(tf.float32)


#layer
W1 = tf.Variable(tf.random_uniform([784,1000],-1,1))
b1 = tf.Variable(tf.random_uniform([1000],-1,1))
L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1,keep_prob)


W2 = tf.Variable(tf.random_uniform([1000,700],-1,1))
b2 = tf.Variable(tf.random_uniform([700],-1,1))
L2 = tf.sigmoid(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2,keep_prob)


W3 = tf.Variable(tf.random_uniform([700,10],-1,1))
b3 = tf.Variable(tf.random_uniform([10],-1,1))
model = tf.matmul(L2,W3)+b3


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=model))

# optimizer = tf.train.GradientDescentOptimizer(0.015).minimize(cost)

#adam
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

batch_size = 50
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)

        _,cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys,keep_prob:0.7})
        total_cost += cost_val

    print("Epoch :",'%04d' % (epoch + 1),
        'Avg. cost=', '{:.3f}'.format(total_cost/total_batch))

is_correct = tf.equal(tf.math.argmax(model,1),tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print("정확도",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prob:1}))






