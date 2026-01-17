import numpy as np
import torch
from .utils import MlatiSessionDataset
from .models import Seq2SeqDecoder
from matplotlib import pyplot as plt

class Result():
    """
    """

    def __init__(self, scores=None):
        """
        """

        if scores is None:
            self._scores = None
        else:
            self._scores = np.asarray(scores).reshape(1, -1)

        return
    
    def merge_with(self, r):
        """
        """

        if self.scores is None:
            self._scores = np.asarray(r.scores).reshape(1, -1)
        else:
            merged = np.vstack([
                self.scores,
                r.scores
            ])
            self._scores = merged
    
    @property
    def scores(self):
        return self._scores

class TimeDependencyAnalysis():
    """
    """

    def __init__(self, ds=None):
        """
        """

        self.ds = ds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.est = None
        self.result = None

        return
    
    def run(
        self,
        n_runs=10,
        window_size=30,
        kernel_size=7,
        train_size=0.8,
        validation_size=0.1,
        lr=0.0001,
        max_iter=500,
        batch_size=32,
        split_seeds=None
        ):
        """
        """

        #
        N, T, U = self.ds.X.shape
        P = self.ds.y.shape[1]
        n_steps = T - window_size
        if split_seeds is None:
            split_seeds = list(range(n_runs))
        n_jobs = n_steps * n_runs
        i_job = 0
        est = Seq2SeqDecoder(U, P, kernel_size=kernel_size, lr=lr, max_iter=max_iter, batch_size=batch_size)
        self.result = Result()

        #
        for i_run in range(n_runs):

            #
            split_seed = split_seeds[i_run]

            #
            ds_train, ds_test = self.ds.random_split([train_size, 1 - train_size], split_seed=split_seed)
            ds_train, ds_valid = ds_train.random_split([1 - validation_size, validation_size], split_seed=split_seed)

            #
            scores = list()
            for i_step in range(n_steps):

                #
                print(f"Working on job {i_job + 1} out of {n_jobs}")

                #
                for ds in (ds_train, ds_valid, ds_test):
                    # X_new = ds.X.copy()[:, i_step:i_step + window_size, :]
                    X_new = ds.X.copy()[:, i_step: i_step + window_size, :]
                    ds.set_X(X_new)

                #
                est._return_to_initial_state()
                est.fit(ds_train, ds_valid, print_info=False)
                score = est.score_r2(ds_test)
                scores.append(score)

                #
                for ds in (ds_train, ds_valid, ds_test):
                    ds.clear_overrides()
                    ds.reset_X()

                #
                i_job += 1            

            #
            result = Result(scores)
            self.result.merge_with(result)

        return
    
    def visualize(self, color="k", t=None, fig=None, figsize=(6, 5)):
        """
        """

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        N, T = self.result.scores.shape
        if t is None:
            t = np.arange(T) + 0.5
        r2_mean = self.result.scores.mean(0)
        r2_std = self.result.scores.std(0)
        ax.plot(t, r2_mean, color=color, alpha=0.7)
        ax.fill_between(t, r2_mean - r2_std, r2_mean + r2_std, color=color, edgecolor="none", alpha=0.3)
        y_min, _ = ax.get_ylim()
        ax.set_ylim([y_min, 1.0])
        ax.set_ylabel(r"$SR^{2}$")
        ax.set_xlabel("Time (in bins)")
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax