import numpy as np
import torch
from src.utils import MlatiSessionDataset
from src.models import Seq2SeqDecoder
from matplotlib import pyplot as plt

class Result():
    """
    """

    _score_X = None
    _score_Z = None
    _score_XZ = None

    def __init__(self,score_X, score_Z, score_XZ):
        """
        """

        self._score_X = round(score_X.item(), 3)
        self._score_Z = round(score_Z.item(), 3)
        self._score_XZ = round(score_XZ.item(), 3)

        return
    
    @property
    def score_X(self):
        return self._score_X
    
    @property
    def score_Z(self):
        return self._score_Z
    
    @property
    def score_XZ(self):
        return self._score_XZ

class VariancePartitioningAnalysis():
    """
    """
    
    def __init__(self, src, ds=None):
        """
        """

        if ds is None:
            self.ds = MlatiSessionDataset(src)
            self.ds.load()
            self.ds.compress() # TODO: Make these transforms specific to the train split (i.e., stop test leakage)
            self.ds.standardize()
        else:
            self.ds = ds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.est = None
        self.result = None

        return
    
    def run(
        self,
        n_runs=3,
        train_size=0.8,
        validation_size=0.1,
        lr=0.0001,
        max_iter=30,
        batch_size=32,
        split_seeds=None
        ):
        """
        """

        #
        target_feature_names = (
            "saccade_amplitude",
            "saccade_endpoints",
            "saccade_startpoints"
        )

        #
        N, T, U = self.ds.X.shape
        K = len(target_feature_names) - 1
        est_X = Seq2SeqDecoder(
            input_size=U,
            output_size=1,
            lr=lr,
            max_iter=max_iter,
            batch_size=batch_size
        )
        est_Z = Seq2SeqDecoder(
            input_size=K,
            output_size=1,
            lr=lr,
            max_iter=max_iter,
            batch_size=batch_size
        )
        est_XZ = Seq2SeqDecoder(
            input_size=U + K,
            output_size=1,
            lr=lr,
            max_iter=max_iter,
            batch_size=batch_size
        )

        #
        self.result = {k: list() for k in target_feature_names}
        n_jobs = len(target_feature_names) * 3 * n_runs
        i_job = 0
        for i_run in range(n_runs):

            # Create fresh subsets
            if split_seeds is None:
                split_seed = i_run + 1
            else:
                split_seed = split_seeds[i_run]
            ds_train, ds_test = self.ds.random_split([train_size, 1 - train_size], split_seed=split_seed)
            ds_train, ds_valid = ds_train.random_split([1 - validation_size, validation_size], split_seed=split_seed)

            #
            for name in target_feature_names:

                #
                print(f"Working on job {i_job + 1} out of {n_jobs} ({(i_job + 1) / n_jobs * 100:.1f}%)")

                # Set target variable
                for ds in (ds_train, ds_valid, ds_test):
                    y_new = getattr(ds, name)
                    ds.clear_overrides()
                    ds.set_y(y_new)

                # Train X-only model
                
                # Fit and eval
                est_X._return_to_initial_state()
                est_X.fit(ds_train, ds_valid, print_info=False)
                score_X = est_X.score(ds_test)
                i_job += 1

                #
                print(f"Working on job {i_job + 1} out of {n_jobs} ({(i_job + 1) / n_jobs * 100:.1f}%)")

                # Train Z-only model
                for ds in (ds_train, ds_valid, ds_test):
                    X_new = list()
                    for attr in target_feature_names:
                        if attr != name:
                            X_new.append(getattr(ds, attr))
                    X_new = np.array(X_new).T
                    X_new = np.repeat(X_new[:, None, :], T, axis=1)
                    ds.set_X(X_new)

                # Fit and eval
                est_Z._return_to_initial_state()
                est_Z.fit(ds_train, ds_valid, print_info=False)
                score_Z = est_Z.score(ds_test)

                # Reset
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_X()

                #
                i_job += 1

                #
                print(f"Working on job {i_job + 1} out of {n_jobs} ({(i_job + 1) / n_jobs * 100:.1f}%)")

                # Train X + Z model
                for ds in (ds_train, ds_valid, ds_test):

                    # Set new inputs and target
                    X_new = list()
                    for attr in target_feature_names:
                        if attr != name:
                            X_new.append(getattr(ds, attr))
                    X_new = np.array(X_new).T
                    X_new = np.repeat(X_new[:, None, :], ds.X.shape[1], axis=1)
                    X_new = np.concatenate([ds.X, X_new], axis=-1)
                    ds.set_X(X_new)

                # Fit and eval
                est_XZ._return_to_initial_state()
                est_XZ.fit(ds_train, ds_valid, print_info=False)
                score_XZ = est_XZ.score(ds_test)

                #
                i_job += 1

                # Reset splits for next loop
                for ds in (ds_train, ds_valid, ds_test):
                    ds.clear_overrides()
                    ds.reset_X()
                    ds.reset_y()

                #
                result = Result(score_X, score_Z, score_XZ)
                self.result[name].append(result)

        return
    
    def visualize(self, figsize=(5, 5)):
        """
        """

        fig, axs = plt.subplots(nrows=len(self.result))
        for i, k in enumerate(self.result.keys()):
            sample_X = np.array([r.score_X for r in self.result[k]])
            sample_Z = np.array([r.score_Z for r in self.result[k]])
            sample_XZ = np.array([r.score_XZ for r in self.result[k]])
            all_samples = (
                sample_X,
                sample_Z,
                sample_XZ
            )
            axs[i].hist(all_samples, bins=30, histtype="stepfilled", color=["C0", "C1", "C2"], alpha=0.3)
            axs[i].set_title(k, fontsize=10)
            y1, y2 = axs[i].get_ylim()
            for s, c in zip(all_samples, ["C0", "C1", "C2"]):
                axs[i].vlines(s.mean(), 0, y2, color=c)
            axs[i].set_ylim([0, y2])
            axs[i].set_ylabel("# of runs")
        axs[-1].set_xlabel("RMSE (pixels)")
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs