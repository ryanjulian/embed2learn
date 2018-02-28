from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

from embed2learn.algos.npo_with_task_embedding import NPTaskEmbedding


class TRPOTaskEmbedding(NPOTaskEmbedding):
    """
    Trust Region Policy Optimization with a Task Embedding
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPOTaskEmbedding, self).__init__(optimizer=optimizer, **kwargs)
