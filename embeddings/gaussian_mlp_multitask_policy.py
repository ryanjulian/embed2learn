import numpy as np
import tensorflow as tf

import itertools

from sandbox.embed2learn.embeddings.base import StochasticEmbedding
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.overrides import overrides
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.embed2learn.embeddings.multitask_policy import StochasticMultitaskPolicy
from sandbox.embed2learn.embeddings.mlp_embedding import MLPEmbedding

class GaussianMLPMultitaskPolicy(MLPEmbedding, StochasticMultitaskPolicy):
    def __init__(
            self,
            name,
            env_spec,
            embedding: StochasticEmbedding,
            task_space,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            std_parameterization='exp'
    ):
        """
        :param env_spec: observation space is a concatenation of task space and vanilla env observation space
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parameterization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        StochasticMultitaskPolicy.__init__(self,env_spec, embedding, task_space)

        with tf.variable_scope(name):
            task_obs_dim = self.task_observation_space.flat_dim
            action_dim = self.action_space.flat_dim
            latent_dim = self.latent_space.flat_dim
            obs_dim = self.observation_space.flat_dim

            # task embedding + plain obs
            latent_obs_dim = latent_dim + obs_dim

            self.task_input = self._embedding._mean_network.input_layer
            self.task_input_var = self.task_input.input_var

            self.env_input = L.InputLayer((None, obs_dim), name="policy_env_input")
            self.env_input_var = self.env_input.input_var

            embed_dist_info_sym = self._embedding.dist_info_sym(
                self.task_input.input_var,
                {
                    self.env_input.input_var: self.env_input.input_var,
                    self.task_input.input_var: self.task_input.input_var
                }
            )
            self.latent_mean_var = embed_dist_info_sym["mean"]
            self.latent_log_std_var = embed_dist_info_sym["log_std"]
            self.latent = L.InputLayer((None, latent_dim), self.latent_mean_var, name="latent_input")
            self._policy_input = L.ConcatLayer((self.latent, self.env_input))

            #init mlp network
            MLPEmbedding.__init__(
                self,
                name=name,
                in_dim=latent_obs_dim,
                latent_dim=action_dim,
                input_layer=self._policy_input,
                hidden_sizes=hidden_sizes,
                learn_std=learn_std,
                init_std=init_std,
                adaptive_std=adaptive_std,
                std_share_network=std_share_network,
                std_hidden_sizes=std_hidden_sizes,
                min_std=min_std,
                std_hidden_nonlinearity=std_hidden_nonlinearity,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                mean_network=mean_network,
                std_network=std_network,
                std_parameterization=std_parameterization)


            dist_info_sym = self.dist_info_sym({
                self.env_input.input_var: self.env_input.input_var,
                self.task_input.input_var: self.task_input.input_var
            }, dict())

            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]
            
            self._task_obs_action_dist = tensor_utils.compile_function(
                inputs=[self.task_input_var, self.env_input.input_var],
                outputs=[mean_var, log_std_var, self.latent_mean_var, self.latent_log_std_var],
            )

            self._latent_obs_action_dist = tensor_utils.compile_function(
                inputs=[self.latent_mean_var, self.env_input.input_var],
                outputs=[mean_var, log_std_var],
            )

    @overrides
    def get_action(self, observation):
        """

        :param observation: task onehot + env observation
        :return: action, dict
        """
        flat_task_obs = self.task_observation_space.flatten(observation)
        flat_task, flat_obs = self.split_observation(flat_task_obs)
        # evaluate embedding
        mean, log_std, latent_mean, latent_log_std = \
            [x[0] for x in self._task_obs_action_dist([flat_task], [flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        latent_info = dict(mean=latent_mean, log_std=latent_log_std)
        return action, dict(mean=mean, log_std=log_std,
                            latent_info=latent_info)

    def get_actions(self, observations):
        raise NotImplementedError()

    @overrides
    def get_params_internal(self, **tags):
        layers = L.get_all_layers(self._output_layers, treat_as_input=self._input_layers)
        layers += L.get_all_layers(self._embedding._output_layers, treat_as_input=self._embedding._input_layers)
        params = itertools.chain.from_iterable(l.get_params(**tags) for l in layers)
        return L.unique(params)

    @overrides
    def get_action_from_latent(self, observation, latent):
        flat_obs = self.observation_space.flatten(observation)
        flat_latent = self.latent_space.flatten(latent)
        # xs = self._latent_obs_action_dist(flat_latent, flat_obs)
        mean, log_std = [x[0] for x in self._latent_obs_action_dist([flat_latent], [flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)


    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        return self.get_reparam_latent_sym(obs_var, action_var, old_dist_info_vars)
        '''
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var
        '''
