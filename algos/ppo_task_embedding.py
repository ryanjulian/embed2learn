import numpy as np

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.optimizers import FirstOrderOptimizer

from sandbox.embed2learn.algos import NPOTaskEmbedding
from sandbox.embed2learn.algos.npo_task_embedding import PGLoss


class PPOTaskEmbedding(NPOTaskEmbedding):
    """
    Proximal Policy Optimization with a Task Embedding

    See https://arxiv.org/abs/1707.06347
    """

    def __init__(self,
                 optimizer_args=dict(batch_size=32, max_epochs=10),
                 **kwargs):
        super(PPOTaskEmbedding, self).__init__(
            pg_loss=PGLoss.CLIP, optimizer=FirstOrderOptimizer, **kwargs)
