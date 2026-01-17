import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .models import RidgeEncoder, MLPEncoder

class NeuralDataProcessor():
    """
    """

    def __init__(self, n_components=3, standardize=False):
        """
        """

        self.scaler = StandardScaler()
        self.decomposer = PCA(n_components=n_components)
        self.n_components = n_components
        self.standardize = standardize

        return
    
    def fit(self, X):
        """
        """

        N, T, U = X.shape
        X_reshaped = X.reshape(N * T, U) # N * T x U (units are columns)
        self.scaler.fit(X_reshaped)
        if self.standardize:
            X_scaled = self.scaler.transform(X_reshaped)
        else:
            X_scaled = X_reshaped
        self.decomposer.fit(X_scaled)

        return self
    
    def transform(self, X):
        """
        """

        N, T, U = X.shape
        X_reshaped = X.reshape(N * T, U) # N * T x U (units are columns)
        if self.standardize:
            X_scaled = self.scaler.transform(X_reshaped)
        else:
            X_scaled = X_reshaped
        X_dec = self.decomposer.transform(X_scaled) # N * T x C components
        X_out = X_dec.reshape(N, T, self.n_components) # N trials x T bins

        return X_out
    
class BehavioralDataProcessor():
    """
    """

    def __init__(self, feature_order=["saccade_direction", "saccade_amplitude", "saccade_startpoints", "saccade_endpoints"]):
        """
        """

        self.scaler = StandardScaler()
        self.feature_order = {f: i for i, f in enumerate(feature_order)}

        return
    
    def fit(self, X):
        """
        """

        self.scaler.fit(X)

        return
    
    def transform(self, X):
        """
        """

        return self.scaler.transform(X)
    
    def drop_feature(self, X, feature_name):
        """
        """

        j = self.feature_order[feature_name]
        X[:, j] = 0.0

        return X
    
    def shuffle_feature(self, X, feature_name, shuffle_index):
        """
        """

        j = self.feature_order[feature_name]
        feature_data = X[shuffle_index, j]
        X[:, j] = feature_data

        return X

class Result():
    """
    """

    def __init__(self, score_loo=None, score_shuffled=None):
        """
        """

        if score_loo is None:
            score_loo = list()
        if score_shuffled is None:
            score_shuffled = list()

        self._score_loo = np.atleast_1d(np.asarray(score_loo))
        self._score_shuffled = np.atleast_1d(np.asarray(score_shuffled))

        return
    
    def merge_with(self, r):
        """
        """

        for attr in ("_score_loo", "_score_shuffled"):
            dst = getattr(self, attr)
            src = getattr(r, attr)
            merged = np.concatenate([dst, src])
            setattr(self, attr, merged)

        return
    
    @property
    def score_loo(self):
        return self._score_loo
    
    @property
    def score_shuffled(self):
        return self._score_shuffled

class SensitivityAnalysisExperiment():
    """
    """

    def __init__(self, ds):
        """
        """

        self.ds = ds
        self.floor = None

        return
    
    def run(
        self,
        train_size=0.8,
        validation_size=0.1,
        l2_penalty=0.001,
        K=100,
        width_scale=1.5,
        lr=0.0005,
        max_iter=1000,
        batch_size=32,
        split_seeds=[1, 2, 3],
        n_components=10,
        model_type="linear",
        ):
        """
        """

        #
        target_feature_names = (
            "saccade_direction",
            "saccade_amplitude",
            "saccade_startpoints",
            "saccade_endpoints"
        )
        self.result = {
            "saccade_direction": Result(),
            "saccade_amplitude": Result(),
            "saccade_startpoints": Result(),
            "saccade_endpoints": Result(),
            "null": list(),
            "baseline": list(),
        }

        #
        _, T, _ = self.ds.X.shape     # Number of time bins (neural activity)
        F = len(target_feature_names) # Number of features  (saccades)
        if model_type == "linear":
            est = RidgeEncoder(
                F=F,
                T=T,
                C=n_components,
                K=K,
                l2_penalty=l2_penalty,
                width_scale=width_scale,
                lr=lr,
                max_iter=max_iter,
                batch_size=batch_size
            )
        elif model_type == "nonlinear": 
            est = MLPEncoder(F=F, T=T, C=n_components, dropout=0.1, hidden_layer_sizes=[256,])
        else:
            raise Exception(f"{model_type} is not a valid model type")

        #
        n_jobs = len(split_seeds) * (2 * F + 2) # 2 tests fits + 1 baseline  and null fits
        i_job = 0

        #
        for i_run, split_seed in enumerate(split_seeds):

            #
            split_seed = int(split_seed)

            # Create the splits
            ds_train, ds_test = self.ds.random_split(
                [train_size, 1 - train_size],
                split_seed=split_seed
            )
            ds_train, ds_valid = ds_train.random_split(
                [1 - validation_size, validation_size],
                split_seed=split_seed
            )

            # Create shuffle index
            # shuffle_seed = split_seed + (abs(hash(target_feature_name)) % 2 ** 31)
            shuffle_seed = split_seed
            rng = np.random.default_rng(shuffle_seed)
            shuffle_index_train = rng.permutation(len(ds_train.X))
            shuffle_index_valid = rng.permutation(len(ds_valid.X))

            # Neural outputs (set once per seed)
            f_X = NeuralDataProcessor(n_components).fit(ds_train.X)
            for ds in (ds_train, ds_valid, ds_test):
                y_new = f_X.transform(ds.X)
                ds.set_y(y_new)

            # Zero and scale kinematic features (using just the training dataset)
            f_y = BehavioralDataProcessor(
                feature_order=[
                    "saccade_direction",
                    "saccade_amplitude",
                    "saccade_startpoints",
                    "saccade_endpoints"
                ]
            )
            y_raw = np.vstack([
                ds_train.saccade_direction,
                ds_train.saccade_amplitude,
                ds_train.saccade_startpoints,
                ds_train.saccade_endpoints
            ]).T
            f_y.fit(y_raw)
            for ds in (ds_train, ds_valid, ds_test):
                y_all = np.vstack([
                    ds.saccade_direction,
                    ds.saccade_amplitude,
                    ds.saccade_startpoints,
                    ds.saccade_endpoints
                ]).T
                X_new = f_y.transform(y_all)
                ds.set_X(X_new)
            
            # Compute baseline performance
            print(f"Working on job {i_job + 1} out of {n_jobs}")
            est._return_to_initial_state()
            est.fit(ds_train, ds_valid, print_info=False)
            baseline = est.score_r2(ds_test)
            self.result["baseline"].append(baseline)
            i_job += 1

            # Shuffle trials (only the training and validation datasets)
            # TODO: This is dumb. I should be shuffling each feature independently, not together.
            for ds, idx in zip([ds_train, ds_valid], [shuffle_index_train, shuffle_index_valid]):
                y_all = np.vstack([
                    ds.saccade_direction[idx],
                    ds.saccade_amplitude[idx],
                    ds.saccade_startpoints[idx],
                    ds.saccade_endpoints[idx],
                ]).T
                X_new = f_y.transform(y_all)
                ds.set_X(X_new)

            # Compute null performance
            print(f"Working on job {i_job + 1} out of {n_jobs}")
            est._return_to_initial_state()
            est.fit(ds_train, ds_valid, print_info=False)
            null = est.score_r2(ds_test)
            self.result["null"].append(null)
            i_job += 1

            # Reset inputs
            for ds in (ds_train, ds_valid, ds_test):
                ds.reset_X()

            # Loop over each feature and run the LOO and permutation protocols
            for target_feature_name in target_feature_names:

                # Behavioral inputs (minus the target input)
                for ds in (ds_train, ds_valid, ds_test):
                    y_all = np.vstack([
                        ds.saccade_direction,
                        ds.saccade_amplitude,
                        ds.saccade_startpoints,
                        ds.saccade_endpoints,
                    ]).T
                    X_new = f_y.transform(y_all)
                    X_new = f_y.drop_feature(X_new, target_feature_name)
                    ds.set_X(X_new)

                # Fit and eval
                print(f"Working on job {i_job + 1} out of {n_jobs}")
                est._return_to_initial_state()
                est.fit(ds_train, ds_valid, print_info=False)
                score_loo = est.score_r2(ds_test)
                i_job += 1

                # Reset inputs
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_X()
                
                # Shuffle target feature for training and validation splits only
                for (ds, idx) in zip([ds_train, ds_valid], [shuffle_index_train, shuffle_index_valid]):
                    y_all = np.vstack([
                        ds.saccade_direction,
                        ds.saccade_amplitude,
                        ds.saccade_startpoints,
                        ds.saccade_endpoints,
                    ]).T
                    X_new = f_y.transform(y_all)
                    X_new = f_y.shuffle_feature(X_new, target_feature_name, idx)
                    ds.set_X(X_new)

                # Preserve trial order for the test split
                y_all = np.vstack([
                    ds_test.saccade_direction,
                    ds_test.saccade_amplitude,
                    ds_test.saccade_startpoints,
                    ds_test.saccade_endpoints,
                ]).T
                X_new = f_y.transform(y_all)
                ds_test.set_X(X_new)
                
                # Fit and eval
                print(f"Working on job {i_job + 1} out of {n_jobs}")
                est._return_to_initial_state()
                est.fit(ds_train, ds_valid, print_info=False)
                score_shuffled = est.score_r2(ds_test)
                i_job += 1

                # Reset inputs
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_X()

                #
                result = Result(score_loo, score_shuffled)
                self.result[target_feature_name].merge_with(result)

        #
        for k in ("baseline", "null"):
            self.result[k] = np.atleast_1d(np.array(self.result[k]))

        return
    
    def visualize(self, figsize=(5, 5)):
        """
        """

        fig, ax = plt.subplots()
        keys = [k for k in list(self.result.keys()) if k not in ("null", "baseline")]
        scores = [self.result[k].score_shuffled.mean() for k in keys]
        index = np.argsort(scores)

        labels = []
        n_keys = len(keys)
        colors = [f"C{i}" for i in range(n_keys)]

        for y, (i, c) in enumerate(zip(index, colors)):
            k = keys[i]
            r = self.result[k]

            ax.hlines(y - 0.15, r.score_loo.mean() - r.score_loo.std(), r.score_loo.mean() + r.score_loo.std(), color=c, alpha=0.7)
            ax.scatter(r.score_loo.mean(), y - 0.15, color=c)

            ax.hlines(y + 0.15, r.score_shuffled.mean() - r.score_shuffled.std(), r.score_shuffled.mean() + r.score_shuffled.std(), color=c, alpha=0.7)
            ax.scatter(r.score_shuffled.mean(), y + 0.15, color=c)

            labels.append(k)

        # Label each saccade feature
        ax.set_yticks(range(n_keys))
        ax.set_yticklabels(labels)

        # Show baseline performance for reference
        y1, y2 = ax.get_ylim()
        x_mean = self.result["baseline"].mean().item()
        x_std = self.result["baseline"].std().item()
        ax.vlines(x_mean, y1, y2, color="k", linestyle=":")
        ax.fill_betweenx([y1, y2], x_mean - x_std, x_mean + x_std, color="k", alpha=0.2, edgecolor="none")

        # Show null performance for reference
        x_mean = self.result["null"].mean().item()
        x_std = self.result["null"].std().item()
        ax.vlines(x_mean, y1, y2, color="r", linestyle=":")
        ax.fill_betweenx([y1, y2], x_mean - x_std, x_mean + x_std, color="r", alpha=0.2, edgecolor="none")
        ax.set_ylim([y1, y2])

        ax.set_xlabel(r"$R^{2}$")
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig, ax
    
def summarize(experiments, figsize=(7, 3)):
    """
    """

    samples = {
        "floor": list(),
        "baseline": list(),
        "saccade_direction_ommission": list(),
        "saccade_amplitude_ommission": list(),
        "saccade_startpoints_ommission": list(),
        "saccade_endpoints_ommission": list(),
        "saccade_direction_shuffle": list(),
        "saccade_amplitude_shuffle": list(),
        "saccade_startpoints_shuffle": list(),
        "saccade_endpoints_shuffle": list(),
    }
    for ex in experiments:
        res = ex.result
        samples["floor"].append(res["null"].mean().item())
        samples["baseline"].append(res["baseline"].mean().item())
        for k in res.keys():
            if k in ("null", "baseline"):
                continue
            ommision_score = res[k].score_loo.mean().item()
            samples[f"{k}_ommission"].append(ommision_score)
            shuffle_score = res[k].score_shuffled.mean().item()
            samples[f"{k}_shuffle"].append(shuffle_score)
    values = list()
    for k, v in samples.items():
        values.append(v)
    values = np.array(values).T

    #
    fig, ax = plt.subplots()
    colors = ["gray", "gray", "C0", "C0", "C0", "C0", "C1", "C1", "C1", "C1"]
    for i in range(values.shape[1]):
        sample = values[:, i]
        x = np.full(sample.size, i) + np.random.normal(loc=0, scale=0.1, size=sample.size)
        ax.scatter(x, sample, color=colors[i], s=20, edgecolor="none", alpha=0.3)
        mean = sample.mean()
        sd = sample.std()
        ax.vlines(i, mean - sd, mean + sd, color=colors[i])
        ax.hlines([mean - sd, mean + sd], i - 0.1, i + 0.1, color=colors[i])
        ax.scatter([i], [mean], marker="o", s=35, color=colors[i], edgecolor="none", zorder=10)

    #
    x1, x2 = ax.get_xlim()
    ax.hlines(np.mean(samples["baseline"]), x1, x2, color="k", linestyle=":")
    ax.set_xlim([x1, x2])

    #
    labels = ("Null", "Baseline", "Dir. (omit)", "Amp. (omit)", "Start (omit)", "End (omit)", "Dir. (shuffle)", "Amp. (shuffle)", "Start (shuffle)", "End (shuffle)")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    axs = np.atleast_1d(ax)
    ax.set_ylabel(r"$R^{2}$")
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()

    return fig, axs
