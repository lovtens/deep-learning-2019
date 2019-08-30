#2014000082
import tensorflow as tf
import numpy as np


sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}


dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) -1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

X = tf.placeholder(tf.int32, [None,sequence_length])
Y = tf.placeholder(tf.int32, [None,sequence_length])
x_one_hot = tf.one_hot(X,num_classes)



cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,x_one_hot,initial_state=initial_state,dtype=tf.float32)





x_for_fc = tf.reshape(outputs,[-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size,sequence_length,num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
# train == optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# prediction == output_argmax
output_argmax = tf.argmax(outputs,axis=2)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
                l, _ = sess.run([loss,optimizer],feed_dict={X:x_data,Y:y_data})
                result = sess.run(output_argmax,feed_dict={X:x_data})

                # print char using dic
                result_str = [idx2char[c] for c in np.squeeze(result)]

                print(i,"loss:",l,"prediction:",''.join(result_str))
                


