import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .model_interfaces import LinearEncoder1, LinearEncoder2, LinearEncoder3, NonlinearEncoder

class NeuralDataTransformer():
    """
    Scales (z-scores) and decomposes neural activity
    """

    def __init__(self, n_components=5, scale=True):
        """
        """

        self.scaler = StandardScaler()
        self.decomposer = PCA(n_components=n_components)
        self.n_components = n_components
        self.scale = scale

        return
    
    def fit(self, X):
        """
        """

        N, T, C = X.shape
        X_reshaped = X.reshape(N * T, C) # N * T x C (units are "channels")
        self.scaler.fit(X_reshaped)
        if self.scale:
            X_scaled = self.scaler.transform(X_reshaped)
        else:
            X_scaled = X_reshaped
        self.decomposer.fit(X_scaled)

        return self
    
    def transform(self, X):
        """
        """

        N, T, C = X.shape
        X_reshaped = X.reshape(N * T, C) # N * T x C_in
        if self.scale:
            X_scaled = self.scaler.transform(X_reshaped)
        else:
            X_scaled = X_reshaped
        X_dec = self.decomposer.transform(X_scaled) # N * T x C_out
        X_out = X_dec.reshape(N, T, self.n_components) # N trials x T bins

        return X_out
    
class KinematicFeatureTransformer():
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

        #
        X_out = self.scaler.transform(X)

        # Override standardization of saccade direction
        j = self.feature_order["saccade_direction"]
        X_override = np.copy(X[:, j])
        X_override[X_override == 0] = -1
        X_out[:, j] = X_override

        return X_out
    
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

    def __init__(self, feature_name, r2_drop=None, r2_shuffle=None, baseline=None, null=None):
        """
        """

        #
        self._feature_name = feature_name

        #
        if r2_drop is None:
            r2_drop = list()
        if r2_shuffle is None:
            r2_shuffle = list()
        if baseline is None:
            baseline = list()
        if null is None:
            null = list()

        #
        self._r2_drop = np.atleast_1d(np.asarray(r2_drop))
        self._r2_shuffle = np.atleast_1d(np.asarray(r2_shuffle))
        self._baseline = np.atleast_1d(np.asarray(baseline))
        self._null = np.atleast_1d(np.asarray(null))

        return
    
    def merge_with(self, r):
        """
        """

        for attr in ("_r2_drop", "_r2_shuffle", "_baseline", "_null"):
            dst = getattr(self, attr)
            src = getattr(r, attr)
            merged = np.concatenate([dst, src])
            setattr(self, attr, merged)

        return
    
    @property
    def feature_name(self):
        return self._feature_name
    
    @property
    def r2_drop(self):
        return self._r2_drop
    
    @property
    def r2_shuffle(self):
        return self._r2_shuffle
    
    @property
    def baseline(self):
        return self._baseline

    @property
    def null(self):
        return self._null

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
        n_runs=3,
        train_size=0.8,
        validation_size=0.1,
        l2_penalty=0.001,
        K=100,
        width_scale=1.5,
        lr=0.0005,
        max_iter=1000,
        batch_size=32,
        split_seeds=None,
        n_components=10,
        model_type="linear1",
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
        result_list = list()

        #
        _, T, _ = self.ds.X.shape     # Number of time bins (neural activity)
        F = len(target_feature_names) # Number of features  (saccades)

        # Ridge regression with raised cosine time basis
        if model_type == "linear1":
            est = LinearEncoder1(
                F=F, T=T, C=n_components, K=K,
                l2_penalty=l2_penalty, width_scale=width_scale,
                lr=lr, max_iter=max_iter, batch_size=batch_size
            )

        # Ridge regression with identity time basis
        elif model_type == "linear2":
            est = LinearEncoder2()
        
        #
        elif model_type == "linear3":
            est = LinearEncoder3()

        # MLP regressor
        elif model_type == "nonlinear": 
            est = NonlinearEncoder(F=F, T=T, C=n_components, dropout=0.1, hidden_layer_sizes=[256,])

        #
        else:
            raise Exception(f"{model_type} is not a valid model type")

        #
        n_jobs = n_runs * (2 * F + 2) # 2 tests fits + 1 baseline  and null fits
        i_job = 0

        #
        if split_seeds is None:
            split_seeds = list(range(n_runs))
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
            tf_X = NeuralDataTransformer(n_components).fit(ds_train.X)
            for ds in (ds_train, ds_valid, ds_test):
                X_out = tf_X.transform(ds.X)
                ds.set_y(X_out)

            # Zero and scale kinematic features (using just the training dataset)
            tf_y = KinematicFeatureTransformer(
                feature_order=[
                    "saccade_direction",
                    "saccade_amplitude",
                    "saccade_startpoints",
                    "saccade_endpoints"
                ]
            )
            y_in = np.vstack([
                ds_train.saccade_direction,
                ds_train.saccade_amplitude,
                ds_train.saccade_startpoints,
                ds_train.saccade_endpoints
            ]).T
            tf_y.fit(y_in)
            for ds in (ds_train, ds_valid, ds_test):
                y_in = np.vstack([
                    ds.saccade_direction,
                    ds.saccade_amplitude,
                    ds.saccade_startpoints,
                    ds.saccade_endpoints
                ]).T
                y_out = tf_y.transform(y_in)
                ds.set_X(y_out)
            
            # Compute baseline performance
            print(f"Working on job {i_job + 1} out of {n_jobs}")
            est._return_to_initial_state()
            est.fit(ds_train, ds_valid, print_info=False)
            r2_baseline = est.score_r2(ds_test)
            i_job += 1

            # Shuffle trials (only the training and validation datasets)
            # TODO: This is dumb. I should be shuffling each feature independently, not together.
            for ds, idx in zip([ds_train, ds_valid], [shuffle_index_train, shuffle_index_valid]):
                y_in = np.vstack([
                    ds.saccade_direction[idx],
                    ds.saccade_amplitude[idx],
                    ds.saccade_startpoints[idx],
                    ds.saccade_endpoints[idx],
                ]).T
                y_out = tf_y.transform(y_in)
                ds.set_X(y_out)

            # Compute null performance
            print(f"Working on job {i_job + 1} out of {n_jobs}")
            est._return_to_initial_state()
            est.fit(ds_train, ds_valid, print_info=False)
            r2_null = est.score_r2(ds_test)
            i_job += 1

            # Reset inputs
            for ds in (ds_train, ds_valid, ds_test):
                ds.reset_X()

            # Loop over each feature and run the LOO and permutation protocols
            for target_feature_name in target_feature_names:

                # Behavioral inputs (minus the target input)
                for ds in (ds_train, ds_valid, ds_test):
                    y_in = np.vstack([
                        ds.saccade_direction,
                        ds.saccade_amplitude,
                        ds.saccade_startpoints,
                        ds.saccade_endpoints,
                    ]).T
                    y_out = tf_y.transform(y_in)
                    y_out = tf_y.drop_feature(y_out, target_feature_name)
                    ds.set_X(y_out)

                # Fit and eval
                print(f"Working on job {i_job + 1} out of {n_jobs}")
                est._return_to_initial_state()
                est.fit(ds_train, ds_valid, print_info=False)
                r2_drop = est.score_r2(ds_test)
                i_job += 1

                # Reset inputs
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_X()
                
                # Shuffle target feature for training and validation splits only
                for (ds, idx) in zip([ds_train, ds_valid], [shuffle_index_train, shuffle_index_valid]):
                    y_in = np.vstack([
                        ds.saccade_direction,
                        ds.saccade_amplitude,
                        ds.saccade_startpoints,
                        ds.saccade_endpoints,
                    ]).T
                    y_out = tf_y.transform(y_in)
                    y_out = tf_y.shuffle_feature(y_out, target_feature_name, idx)
                    ds.set_X(y_out)

                # Preserve trial order for the test split
                y_in = np.vstack([
                    ds_test.saccade_direction,
                    ds_test.saccade_amplitude,
                    ds_test.saccade_startpoints,
                    ds_test.saccade_endpoints,
                ]).T
                y_out = tf_y.transform(y_in)
                ds_test.set_X(y_out)
                
                # Fit and eval
                print(f"Working on job {i_job + 1} out of {n_jobs}")
                est._return_to_initial_state()
                est.fit(ds_train, ds_valid, print_info=False)
                r2_shuffle = est.score_r2(ds_test)
                i_job += 1

                # Reset inputs
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_X()

                #
                result = Result(target_feature_name, r2_drop, r2_shuffle, r2_baseline, r2_null)
                result_list.append(result)

        #
        self.result = {}
        for k in target_feature_names:
            target_results = [r for r in result_list if r.feature_name == k]
            merged = Result(k)
            for r in target_results:
                merged.merge_with(r)
            self.result[k] = merged

        return
    
    def visualize(self, figsize=(5, 3), ax=None):
        """
        Drop-in replacement for your current visualize().

        Plots mean±SD for r2_drop (circle) and r2_shuffle (diamond) per feature,
        ordered by mean r2_shuffle, with correctly aligned y-ticks/labels.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # Create axis if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_ax = True
        else:
            fig = None
            created_ax = False

        # ---- Order features by shuffle performance (low->high) ----
        keys_sorted = sorted(
            list(self.result.keys()),
            key=lambda k: float(np.mean(self.result[k].r2_shuffle))
        )

        # ---- Color map (stable across sorted order) ----
        colors = [f"C{i}" for i in range(len(keys_sorted))]
        cmap = {k: c for k, c in zip(keys_sorted, colors)}

        # ---- Plot ----
        labels = []
        for y, k in enumerate(keys_sorted):
            r = self.result[k]
            c = cmap[k]

            drop_mean = float(np.mean(r.r2_drop))
            drop_std  = float(np.std(r.r2_drop))
            shuf_mean = float(np.mean(r.r2_shuffle))
            shuf_std  = float(np.std(r.r2_shuffle))

            # drop (lower row)
            ax.hlines(y - 0.2, drop_mean - drop_std, drop_mean + drop_std, color=c, alpha=0.5)
            ax.scatter(drop_mean, y - 0.2, color=c, zorder=10)

            # shuffle (upper row)
            ax.hlines(y + 0.2, shuf_mean - shuf_std, shuf_mean + shuf_std, color=c, alpha=0.5)
            ax.scatter(shuf_mean, y + 0.2, color=c, marker="D", zorder=10)

            labels.append(k)

        # ---- Y ticks/labels (aligned to row centers) ----
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([k.split("_", 1)[1] if "_" in k else k for k in labels])

        # ---- Reference lines (baseline + null) ----
        # Keep your original behavior (using saccade_direction container)
        y1, y2 = ax.get_ylim()

        base_mean = float(np.mean(self.result["saccade_direction"].baseline))
        base_std  = float(np.std(self.result["saccade_direction"].baseline))
        ax.vlines(base_mean, y1, y2, color="k", linestyle=":")
        ax.fill_betweenx([y1, y2], base_mean - base_std, base_mean + base_std,
                        color="k", alpha=0.2, edgecolor="none")

        null_mean = float(np.mean(self.result["saccade_direction"].null))
        null_std  = float(np.std(self.result["saccade_direction"].null))
        ax.vlines(null_mean, y1, y2, color="r", linestyle=":")
        ax.fill_betweenx([y1, y2], null_mean - null_std, null_mean + null_std,
                        color="r", alpha=0.2, edgecolor="none")

        ax.set_ylim([y1, y2])

        # ---- Labels ----
        ax.set_xlabel(r"$R^{2}$")

        # ---- Layout ----
        if created_ax:
            fig.tight_layout()

        return fig, ax, cmap

def summarize(experiments, example, figsize=(9, 3)):
    """
    """

    samples = {k: list() for k in experiments[0].result.keys()}
    for ex in experiments:
        result = ex.result
        for k in result.keys():
            x = result[k].baseline.mean()
            y = result[k].r2_drop.mean()
            z = 1 - y / x
            samples[k].append(round(float(z), 6))

    #
    fig, axs = plt.subplots(ncols=2)

    #
    _, _, cmap = example.visualize(ax=axs[0])

    #
    for k, v in samples.items():
        axs[1].hist(
            v,
            bins=25,
            range=(-0.1, 0.4),
            color=cmap[k],
            alpha=0.3,
            
        )
        label = k.split("_")[1]
        x0 = np.mean(v)
        axs[1].vlines(x0, 0, 3.1, color=cmap[k], label=label)


    #
    axs[1].set_ylabel("# of populations")
    # axs[1].set_ylim([0, 3.1])

    #
    axs[1].set_xlabel("Frac. of explainable variance")
    plt.legend()
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()

    return fig, axs