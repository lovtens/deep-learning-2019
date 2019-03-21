import tensorflow as tf 

x_data = [1,2,3]
y_data = [1,2,3]


W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W * X + b



cost = tf.reduce_mean(tf.square(hypothesis -Y))
#local minima 크다고 좋은게 아니고 나쁜것도 아니고
#hyper parameter(learning rate 등)가 중요
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(30): #epoch 오버피팅하지 않게 조절
        _, cost_val = sess.run([train_op,cost],feed_dict={X:x_data, Y:y_data})
        # 얼마나 잘학습하고 있는지 cost출력해서 체크
        print(step,cost_val, sess.run(W), sess.run(b))


    #학습완료 후 모델 테스트
    print("X : 6, Y : ",sess.run(hypothesis,feed_dict={X:6}))
    print("X : 2.7, Y : ",sess.run(hypothesis,feed_dict={X:2.7}))







