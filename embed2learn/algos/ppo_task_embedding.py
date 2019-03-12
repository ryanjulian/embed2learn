from garage.tf.optimizers import FirstOrderOptimizer

from embed2learn.algos import NPOTaskEmbedding
from embed2learn.algos.npo_task_embedding import PGLoss


class PPOTaskEmbedding(NPOTaskEmbedding):
    """
    Proximal Policy Optimization with a Task Embedding

    See https://arxiv.org/abs/1707.06347
    """

    def __init__(self,
                 step_size=0.2,
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
            step_size=step_size,
            optimizer=FirstOrderOptimizer,
            optimizer_args=optimizer_args,
            inference_optimizer=FirstOrderOptimizer,
            inference_optimizer_args=inference_optimizer_args,
            **kwargs)
