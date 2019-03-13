import argparse

from akro.tf import Box
from garage.envs import EnvSpec
from garage.misc import logger
# import ipdb
import numpy as np
import tensorflow as tf

from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.policies import GaussianMLPMultitaskPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int)
args = parser.parse_args()

with tf.Session() as sess:
    logger.set_tensorboard_dir("../../data/local/test_fixture/temp{}".format(
        args.i))

    task_space = Box(low=np.array([0, 0]), high=np.array([1, 1]))
    latent_space = Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))
    embed_spec = EmbeddingSpec(task_space, latent_space)

    obs_space = Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]))
    act_space = Box(
        low=np.array([-2, -2, -2, -2, -2]), high=np.array([2, 2, 2, 2, 2]))
    env_spec = EnvSpec(obs_space, act_space)

    e = GaussianMLPEmbedding(embed_spec, std_share_network=True)
    p = GaussianMLPMultitaskPolicy(
        env_spec=env_spec,
        task_space=task_space,
        embedding=e,
        std_share_network=True)

    my_task = task_space.new_tensor_variable(name="my_task", extra_dims=1)
    my_obs = obs_space.new_tensor_variable(name="my_obs", extra_dims=1)
    with tf.name_scope("build_opt"):
        dist_info = e.dist_info_sym(my_task, name="e_dist_info")
        p_dist_info = p.dist_info_sym(my_task, my_obs, name="p_dist_info")

    with tf.name_scope("test_fixture"):
        a = tf.exp(e.latent)
        b = tf.exp(e._latent_std_param)

    sess.run(tf.global_variables_initializer())

    t = task_space.sample()
    o = obs_space.sample()
    to = np.concatenate([t, o], axis=0)
    p.get_action(to)
    z = latent_space.sample()
    p.get_action_from_latent(o, z)

    logger.dump_tensorboard()
    # ipdb.set_trace()

    print("done!")
