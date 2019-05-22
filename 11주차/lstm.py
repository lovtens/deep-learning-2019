#2014000082
import tensorflow as tf
import numpy as np


sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

idx2char = list(set(sentence))
# print(idx2char)
char2idx = {w: i for i, w in enumerate(idx2char)}

dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
sequence_length = 10
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sentence]

x_data = []
y_data = []

for i in range(0,len(sentence) - sequence_length):
        x_str = sentence[i:i+sequence_length]
        y_str = sentence[i+1:i+1+sequence_length]
        print(i,":",x_str,'->',y_str,end='\n')

        x = [char2idx[c] for c in x_str]
        y = [char2idx[c] for c in y_str]

        x_data.append(x)
        y_data.append(y)

batch_size = len(x_data)

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
                results = sess.run(output_argmax,feed_dict={X:x_data})

                for j, result in enumerate(results):
                        print(i,j,''.join([idx2char[t] for t in result]),l)

        # results = sess.run(output_argmax,feed_dict={X:x_data})
        results = sess.run(outputs,feed_dict={X:x_data})
        for j, result in enumerate(results):
                index = np.argmax(result,axis=1)
                # index = result
                if j is 0:
                        print(''.join([idx2char[t] for t in index]),end='')
                else:``
                        print(idx2char[index[-1]],end='')
                


