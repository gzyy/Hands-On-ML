import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
n_epochs = 10
batch_size = 200

#for op in tf.get_default_graph().get_operations():
#    print(op.name)

saver = tf.train.import_meta_graph("./my_reuse_1.ckpt.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")

training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")

for op in (X,y,accuracy,training_op):
    tf.add_to_collection("important_ops",op)

X,y,accuracy,training_op = tf.get_collection("important_ops")

with tf.Session() as sess:
    saver.restore(sess,"./my_reuse_1.ckpt")

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,
                                                y:mnist.test.labels})
        print(epoch,"Test accuracy:",accuracy_val)

    save_path = saver.save(sess,"./my_reuse_final.ckpt")
