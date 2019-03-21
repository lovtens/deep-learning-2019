import tensorflow as tf 

x_data = [[0,0], [0,1], [1,0], [1,1]]
#XOR
y_data = [[0],[1],[1],[0]]


X = tf.placeholder(tf.float32, [None,2])
Y = tf.placeholder(tf.float32, [None,1])

W1 = tf.Variable(tf.random_uniform([2,3],-1.0,1.0))
b1 = tf.Variable(tf.random_uniform([1,3],-1.0,1.0))

W2 = tf.Variable(tf.random_uniform([3,1],-1.0,1.0))
b2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))



logits1 = tf.add(tf.matmul(X,W1),b1)
output1 = tf.nn.sigmoid(logits1)

logits2 = tf.add(tf.matmul(output1,W2),b2)
output = tf.nn.sigmoid(logits2)

cost = tf.reduce_mean(tf.square(output -Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
       for x, y in zip(x_data,y_data):
           _, cost_val = sess.run([train_op,cost],feed_dict={X:[x],Y:[y]})
        #    print(step,cost_val,sess.run(W1),sess.run(b1),sess.run(W2),sess.run(b2))

    print(sess.run(output,feed_dict={X:x_data}))






