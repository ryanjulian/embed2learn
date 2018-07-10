import numpy as np

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.optimizers import FirstOrderOptimizer

from sandbox.embed2learn.algos import NPOTaskEmbedding
from sandbox.embed2learn.algos.npo_task_embedding import PGLoss

# TODO: FirstOrderOptimizer seems to be reshaping inputs inappropriately


class PPOTaskEmbedding(NPOTaskEmbedding):
    """
    Proximal Policy Optimization with a Task Embedding

    See https://arxiv.org/abs/1707.06347
    """

    def __init__(self,
                 optimizer_args=dict(
                     batch_size=32,
                     max_epochs=10,
                     tf_optimizer_args=dict(learning_rate=1e-3)),
                 inference_optimizer_args=dict(
                     batch_size=32,
                     max_epochs=5,
                     tf_optimizer_args=dict(learning_rate=1e-3)),
                 **kwargs):
        super(PPOTaskEmbedding, self).__init__(
            pg_loss=PGLoss.CLIP,
            optimizer=FirstOrderOptimizer,
            optimizer_args=optimizer_args,
            inference_optimizer=FirstOrderOptimizer,
            inference_optimizer_args=inference_optimizer_args,
            **kwargs)
