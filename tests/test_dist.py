import numpy as np
import tensorflow as tf
from garage.tf.distributions import DiagonalGaussian as RockyDiagonalGaussian
from sandbox.embed2learn.distributions import DiagonalGaussian as ZhanpengDiagonalGaussian
from sandbox.embed2learn.core.networks import MLP


def test_dg():
#     means = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     stds = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
#     xs = np.array([1, 2, 2, 1, 1, 2, 2, 1, 1, 2])
#
#     means_2 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0])
#     stds_2 = np.array([1.05, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15])
#
#     dist_info = dict(mean=means, log_std=stds)
#     dist_info_2 = dict(mean=means_2, log_std=stds_2)
#
#     dist = RockyDiagonalGaussian(10)
#
#     log_likelihood = dist.log_likelihood(xs=xs, dist_info=dist_info)
#     kl = dist.kl(dist_info, dist_info_2)
#     print(log_likelihood, kl)
#
#     # Zhanpeng's
#     means_ph = tf.placeholder(tf.float32, shape=(10,))
#     std_ph = tf.placeholder(tf.float32, shape=(10,))
#
#     means_ph_2 = tf.placeholder(tf.float32, shape=(10,))
#     std_ph_2 = tf.placeholder(tf.float32, shape=(10,))
#
#     value_ph = tf.placeholder(tf.float32, shape=(10,))
#     latent_sample_size_ph = tf.placeholder(tf.int32, shape=None)
#
#     new_dist = ZhanpengDiagonalGaussian(means_ph, std_ph, 10)
#     new_dist_2 = ZhanpengDiagonalGaussian(means_2, std_ph_2, 10)
#
#     log_prob = new_dist.log_prob(value=value_ph)

    mlp = MLP(input_dim=3, output_dim=2)
    ph = tf.placeholder(tf.float32, shape=[None, 3], name='new_ph')
    out = mlp.feedforward(input_ph=ph, reuse=True,)

    sess = tf.Session()

    tf.summary.FileWriter('/home/zhanpenghe/Desktop/train', sess.graph)

    sess.run(tf.global_variables_initializer())

    feed_dict = {
        mlp.input_ph: np.array([[0, 1, 2]])
    }

    feed_dict_2 = {
        ph: np.array([[0, 1, 2]])
    }

    loss = mlp.output_op
    adam = tf.train.AdamOptimizer(0.003).minimize(loss)
    sess.run(tf.global_variables_initializer())
    print(sess.run(mlp.output_op, feed_dict=feed_dict))
    print(sess.run(out, feed_dict=feed_dict_2))

    sess.run(adam, feed_dict=feed_dict)

    print(sess.run(mlp.output_op, feed_dict=feed_dict))
    print(sess.run(out, feed_dict=feed_dict_2))
#     feed_dict = {
#         means_ph: means,
#         std_ph: stds,
#         means_ph_2: means_2,
#         std_ph_2: stds_2,
#         value_ph: xs,
#         latent_sample_size_ph: 1,
#     }
#
#
#     log_likelihood, kl = sess.run([log_prob, new_dist.kl_divergence(new_dist_2)], feed_dict=feed_dict)
#
#     # print(log_likelihood, kl)
#
#     sample_op_1 = new_dist.sample(sample_shape=(1,), seed=1, name='sample_1')
#     sample_op_2 = new_dist.sample(sample_shape=(2,), seed=1, name='sample_2')
#
#     # print(sample_op_1, sample_op_2)
#
#     for i in range(10):
#         sampled_1 = sess.run(sample_op_1, feed_dict=feed_dict)
#         sampled_2 = sess.run(sample_op_2, feed_dict=feed_dict)
#         # print(sampled_1, sampled_2)
#
#
#
#
test_dg()
