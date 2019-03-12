import unittest

from akro.tf import Box
from garage.envs.env_spec import EnvSpec
import numpy as np
import tensorflow as tf

from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.policies import GaussianMLPMultitaskPolicy


class TestStdShareNetwork(unittest.TestCase):
    def test_init_std(self):
        sess = tf.Session()
        sess.__enter__()


        task_space = Box(
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32))
        latent_space = Box(
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32))
        embed_spec = EmbeddingSpec(task_space, latent_space)
        embedding = GaussianMLPEmbedding(
            name="embedding",
            embedding_spec=embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=1.0,
            max_std=2.0,
        )

        std_parameterization = "exp"
        observation_space = Box(np.full(100, 0.0), np.full(100, 100.0))
        action_space = Box(np.full(10, 0.0), np.full(10, 10.0))
        env_spec = EnvSpec(observation_space, action_space)
        policy = GaussianMLPMultitaskPolicy(
            name="policy",
            env_spec=env_spec,
            task_space=task_space,
            embedding=embedding,
            hidden_sizes=(200, 100),
            std_share_network=True,
            # max_std=10.0,
            init_std=6.0,
            std_parameterization=std_parameterization,
        )

        sess.run(tf.global_variables_initializer())

        z = latent_space.sample()
        # z = np.ones_like(latent_space.low)
        print("|z| = {}".format(np.linalg.norm(z)))
        # z = z / np.linalg.norm(z)

        o = observation_space.sample()
        # o = np.ones_like(observation_space.low)
        print("|o| = {}".format(np.linalg.norm(o)))
        # o = o / np.linalg.norm(o)

        a, info = policy.get_action_from_latent(z, o)

        log_stds = info["log_std"]
        if std_parameterization == "exp":
            stds = np.exp(log_stds)
        elif std_parameterization == "softplus":
            stds = np.log(1. + np.exp(log_stds))
        else:
            raise NotImplementedError

        print("log_stds = {}".format(log_stds))
        print("stds = {}".format(stds))
        print("mean(stds) = {}".format(np.mean(stds)))
        print("std(stds) = {}".format(np.std(stds)))

        assert np.allclose(stds, 1.0), "stds: {}".format(stds)
