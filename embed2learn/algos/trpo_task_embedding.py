from enum import Enum
from enum import unique

from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import PenaltyLbfgsOptimizer

from embed2learn.algos import NPOTaskEmbedding
from embed2learn.algos.npo_task_embedding import PGLoss


@unique
class KLConstraint(Enum):
    HARD = "hard"
    SOFT = "soft"


class TRPOTaskEmbedding(NPOTaskEmbedding):
    """
    Trust Region Policy Optimization with a Task Embedding
    """

    def __init__(self, kl_constraint=KLConstraint.HARD, max_kl_step=0.01,
        **kwargs):
        if kl_constraint == KLConstraint.HARD:
            optimizer = ConjugateGradientOptimizer
        elif kl_constraint == KLConstraint.SOFT:
            optimizer = PenaltyLbfgsOptimizer
        else:
            raise NotImplementedError("Unknown KLConstraint")

        super(TRPOTaskEmbedding, self).__init__(
            pg_loss=PGLoss.VANILLA, optimizer=optimizer, **kwargs)
