import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_outputs = 10

scale = 0.001 # L1 regularization hyperparameter
learning_rate = 0.01
n_epochs = 20
batch_size = 200

X = tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X,n_hidden1,activation=tf.nn.relu,name="hidden1")
    logits = tf.layers.dense(hidden1,n_outputs,name="outputs")

W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    base_loss = tf.reduce_mean(xentropy,name="avg_xentropy")
    reg_losses = tf.reduce_sum(tf.abs(W1))+tf.reduce_sum(tf.abs(W2))
    loss = tf.add(base_loss,scale*reg_losses,name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
#saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,
                                                y:mnist.test.labels})
        print(epoch,"Test accuracy:",accuracy_val)