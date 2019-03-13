from garage.experiment import LocalRunner
from garage.tf.plotter import Plotter

from embed2learn.samplers import TaskEmbeddingSampler
from embed2learn.samplers.task_embedding_sampler import rollout


class TaskEmbeddingRunner(LocalRunner):

    def setup(self, algo, env, batch_size, max_path_length):
        n_envs = batch_size // max_path_length
        return super().setup(
            algo=algo,
            env=env,
            sampler_cls=TaskEmbeddingSampler,
            sampler_args=dict(n_envs=n_envs))

    def train(self,
          n_epochs,
          n_epoch_cycles=1,
          plot=False,
          store_paths=False,
          pause_for_plot=False):
        return super().train(
            n_epochs,
            n_epoch_cycles=n_epoch_cycles,
            plot=plot,
            store_paths=store_paths,
            pause_for_plot=pause_for_plot)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy, rollout=rollout)
            self.plotter.start()
