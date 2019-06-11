from tensorflow.python.ops import math_ops  
import tensorflow as tf

input = tf.constant([[-5, 10000], [150, 10], [5, 100]])  
input_data = tf.constant([3.1])
boundaries = [4.6, 8.0, 10.0, 15.9]

a=math_ops.bucketize(input_data,boundaries) 

sess = tf.Session()
print("result",sess.run(a))