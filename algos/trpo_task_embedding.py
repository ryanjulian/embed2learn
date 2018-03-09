from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

from sandbox.embed2learn.algos.npo_task_embedding import NPOTaskEmbedding


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
