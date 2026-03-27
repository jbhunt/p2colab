import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .models import RNNDecoder, LinearDecoder, MLPDecoder
from .datasets import PsuedoSessionDataset
from .utils import NeuralActivityProcessor

class Result:
    """
    Store out-of-fold predictions and compute pooled R^2 by feature and step.
    """

    def __init__(self, pad=5000):
        """
        """

        self._feature = []
        self._step = []
        self._score = None
        self._y_pred = None
        self._y_true = None
        self.pad = pad

        return

    def cache(self, feature, step, y_true, y_pred):
        """
        Store one set of held-out predictions for a single fold / feature / step.
        """

        self._feature.append(feature)
        self._step.append(step)

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.size != y_pred.size:
            raise ValueError("y_true and y_pred must have same length")

        if y_true.size > self.pad:
            raise ValueError(
                f"y_true length ({y_true.size}) exceeds pad ({self.pad}). "
                "Increase pad or store variable-length arrays differently."
            )

        pad_width = self.pad - y_true.size
        y_true_pad = np.pad(y_true, (0, pad_width), mode="constant", constant_values=np.nan)
        y_pred_pad = np.pad(y_pred, (0, pad_width), mode="constant", constant_values=np.nan)

        if self._y_true is None:
            self._y_true = y_true_pad.reshape(1, -1)
            self._y_pred = y_pred_pad.reshape(1, -1)
        else:
            self._y_true = np.vstack([self._y_true, y_true_pad])
            self._y_pred = np.vstack([self._y_pred, y_pred_pad])

        return

    def process(self):
        """
        Compute pooled R^2 for each unique feature and time step
        """

        feature = np.array(self._feature)
        step = np.array(self._step)

        unique_pairs = []
        for k, s in zip(feature, step):
            pair = (k, s)
            if pair not in unique_pairs:
                unique_pairs.append(pair)

        out_feature = []
        out_step = []
        out_score = []

        for k, s in unique_pairs:
            m = (feature == k) & (step == s)

            y_true = self._y_true[m].ravel()
            y_pred = self._y_pred[m].ravel()

            keep = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true = y_true[keep]
            y_pred = y_pred[keep]

            rss = np.sum((y_pred - y_true) ** 2)
            tss = np.sum((y_true - y_true.mean()) ** 2)

            if tss == 0:
                r2 = np.nan
            else:
                r2 = 1 - rss / tss

            out_feature.append(k)
            out_step.append(s)
            out_score.append(r2)

        self._feature = np.array(out_feature)
        self._step = np.array(out_step)
        self._score = np.array(out_score)

    @property
    def score(self):
        return self._score

    @property
    def feature(self):
        return np.array(self._feature)

    @property
    def step(self):
        return np.array(self._step)

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred
    
class SingleDecodingExperiment():
    """
    """

    ks = (
        "saccade_direction",
        "saccade_amplitude",
        "saccade_startpoints",
        "saccade_endpoints",
        "saccade_velocity"
    )

    def __init__(
        self,
        ds,
        window_size=1,
        window_stride=1,
        kernel_size=5,
        n_components=3,
        n_splits=10,
        alpha=1.0,
        validation_fraction=0.1,
        lr=0.0001,
        max_iter=1000,
        batch_size=16,
        model_type="linear",
        split_seed=42,
        shuffle_trials=False
        ):
        """
        """

        #
        self.ds = ds
        self.n_components = n_components
        self.n_splits = n_splits
        self.validation_fraction = validation_fraction
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.window_stride = window_stride
        self.result = None
        self.est = None
        self.model_type = model_type
        if self.model_type not in ["rnn", "linear", "mlp"]:
            raise Exception(f"{model_type} is not a supported model type")
        self.split_seed = split_seed
        self.shuffle_trials = shuffle_trials

        # Store timepoint for the right-most bin (for plotting)
        N, T, C = self.ds.X.shape
        left_edges = np.arange(0, T - self.window_size + 1, self.window_stride)
        right_edges = left_edges + self.window_size - 1
        self.t = self.ds.t_X[right_edges]

        return
    
    def run(self, unit_types=["premotor", "visuomotor", "visual"]):
        """
        """

        # Initialize the decoder
        N, T, C = self.ds.filter_X(unit_types=unit_types).shape
        if self.n_components is None:
            input_size = C
        else:
            input_size = self.n_components
        if self.model_type == "rnn":
            self.est = RNNDecoder(
                input_size,
                output_size=1,
                kernel_size=self.kernel_size,
                lr=self.lr,
                max_iter=self.max_iter,
                batch_size=self.batch_size
            )
        elif self.model_type == "linear":
            self.est = LinearDecoder(
                max_iter=self.max_iter,
                alpha=self.alpha
            )
        elif self.model_type == "mlp":
            self.est = MLPDecoder(
                max_iter=self.max_iter,
                lr=self.lr,
                batch_size=self.batch_size
            )

        #
        starts = np.arange(0, T - self.window_size + 1, self.window_stride)
        ends = starts + self.window_size
        window_edges = np.column_stack((starts, ends))
        n_steps = window_edges.shape[0]
        n_jobs = self.n_splits * n_steps * 5
        i_job = 0

        # Get set up for training
        splits = self.ds.kfold_split(k=self.n_splits, validation_fraction=self.validation_fraction, split_seed=self.split_seed)
        self.result = Result()

        # For each train, validation, test split ...
        for i_split in range(self.n_splits):

            # Grab data subsets
            ds_train = splits["train"][i_split]
            ds_test = splits["test"][i_split]
            ds_valid = splits["valid"][i_split]

            # Standardize and decompose neural data
            tf_X = NeuralActivityProcessor(n_components=self.n_components)
            X_train = ds_train.filter_X(unit_types=unit_types)
            X_valid = ds_valid.filter_X(unit_types=unit_types)
            X_test = ds_test.filter_X(unit_types=unit_types)

            # Shuffle trials (optional)
            if self.shuffle_trials:
                for X in [X_train, X_valid, X_test]:
                    np.random.shuffle(X)

            # For each kinematic feature
            for j, k in enumerate(self.ks):

                # Standardize target kinematic feature and assign it as the prediction target
                tf_y = StandardScaler()
                tf_y.fit(getattr(ds_train, k).reshape(-1, 1))
                for ds in [ds_train, ds_valid, ds_test]:
                    y = getattr(ds, k).reshape(-1, 1)
                    y = tf_y.transform(y).flatten()
                    ds.set_y(y)
                
                # Move through time
                for i_step, (start, stop) in enumerate(window_edges):

                    # Slice out window of neural data
                    tf_X.fit(X_train[:, start: stop, :])
                    for ds, X in zip([ds_train, ds_valid, ds_test], [X_train, X_valid, X_test]):
                        X = tf_X.transform(X[:,start: stop, :])
                        ds.set_X(X)

                    #
                    end = "\r" if (i_job + 1) < n_jobs else "\n"
                    message = f"Working on job {i_job + 1} out of {n_jobs}"
                    print(message, end=end)

                    # Fit, evaluate, and track performance
                    self.est.reset()
                    self.est.fit(ds_train, ds_valid, print_info=False)
                    y_pred = self.est.predict(ds_test)
                    y_true = ds_test.y

                    #
                    self.result.cache(
                        feature=k,
                        step=i_step,
                        y_true=y_true,
                        y_pred=y_pred
                    )

                    # Reset datasets
                    for ds in (ds_train, ds_valid, ds_test):
                        ds.reset_X()

                    #
                    i_job += 1

                #
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_y()

        # Compute R^2 for each feature and time step
        self.result.process()

        return
    
    def plot_summary(self):
        """
        """

        fig, ax = plt.subplots()

        for k, c in zip(self.ks, ["C0", "C1", "C2", "C3"]):
            m = self.result.feature == k
            steps = self.result.step[m]
            r2 = self.result.score[m]
            idx = np.argsort(steps)
            ax.plot(self.t[steps[idx]], r2[idx], color=c, label=k)

        ax.set_xlabel("Time")
        ax.set_ylabel(r"$R^2$")
        ax.legend()

        return fig, [ax,]
    
    def plot_example(self, t=None, k="saccade_amplitude", unit_types=["visuomotor", "premotor"]):
        """
        """

        #
        self.est.reset()

        #
        if t is None:
            mask = self.result.feature == k
            i = np.argmax(self.result.score[mask])
            i_t = self.result.step[mask][i]
        else:
            i_t = np.argmin(np.abs(self.ds.t_X - t))

        #
        splits = self.ds.kfold_split(k=self.n_splits, validation_fraction=self.validation_fraction, split_seed=self.split_seed)
        y_pred = list()
        y_true = list()

        # For each train, validation, test split ...
        for i_split in range(self.n_splits):

            # Grab data subsets
            ds_train = splits["train"][i_split]
            ds_test = splits["test"][i_split]
            ds_valid = splits["valid"][i_split]

            # Standardize and decompose neural data
            tf_X = NeuralActivityProcessor(n_components=self.n_components)
            X_train = ds_train.filter_X(unit_types=unit_types)
            X_valid = ds_valid.filter_X(unit_types=unit_types)
            X_test = ds_test.filter_X(unit_types=unit_types)

            #
            tf_y = StandardScaler()
            tf_y.fit(getattr(ds_train, k).reshape(-1, 1))
            for ds in [ds_train, ds_valid, ds_test]:
                y = getattr(ds, k).reshape(-1, 1)
                y = tf_y.transform(y).flatten()
                ds.set_y(y)

            # Slice out window of neural data
            tf_X.fit(X_train[:, i_t, :][:, None, :])
            for ds, X in zip([ds_train, ds_valid, ds_test], [X_train, X_valid, X_test]):
                X = tf_X.transform(X[:, i_t, :][:, None, :])
                ds.set_X(X)

            # Fit, evaluate, and track performance
            self.est.reset()
            self.est.fit(ds_train, ds_valid, print_info=False)
            y_pred = np.concatenate([
                y_pred,
                tf_y.inverse_transform(self.est.predict(ds_test).reshape(-1, 1)).flatten()
            ])
            y_true = np.concatenate([
                y_true,
                tf_y.inverse_transform(ds_test.y.reshape(-1, 1)).flatten()
            ])

        #
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, marker=".", color="k", edgecolor="none", alpha=0.5, s=15)
        ax.set_aspect("equal")
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.set_ylim()
        xylim = [
            min(x1, y1),
            max(x2, y2)
        ]
        ax.plot(xylim, xylim, color="k", linestyle=":")
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_xlabel(r"$y_{true}$")
        ax.set_ylabel(r"$y_{pred.}$")

        return fig, [ax,]
        
class AllDecodingExperiments:
    """
    """

    def __init__(self, sessions, n_runs=10, **kwargs):
        """
        """

        self.kwargs = {
            "window_size": 1,
            "window_stride": 1,
            "kernel_size": 5,
            "n_components": 3,
            "n_splits": 10,
            "alpha": 1.0,
            "validation_fraction": 0.1,
            "lr": 0.0001,
            "max_iter": 1000,
            "batch_size": 16,
            "model_type": "linear"
        }
        self.kwargs.update(kwargs)
        self.n_runs = n_runs
        self.sessions = sessions
        self.experiments = None

        return

    def run(self):
        """
        """

        pseudosession = PsuedoSessionDataset(
            sessions=self.sessions,
            build=False
        )
        self.experiments = {
            ("single", "visual", "unshuffled"): list(),
            ("single", "premotor", "unshuffled"): list(),
            ("single", "visual", "shuffled"): list(),
            ("single", "premotor", "shuffled"): list(),
            ("pseudo", "visual", "unshuffled"): list(),
            ("pseudo", "premotor", "unshuffled"): list(),
            ("pseudo", "visual", "shuffled"): list(),
            ("pseudo", "premotor", "shuffled"): list(),
        }
        n_jobs = (4 * len(self.sessions)) + (4 * self.n_runs)
        i_job = 0
        for ex_key in self.experiments.keys():

            #
            if ex_key[1] == "visual":
                unit_types = ["visual"]
            elif ex_key[1] == "premotor":
                unit_types = ["visuomotor", "premotor"]
            else:
                raise Exception()
            shuffle_trials = True if ex_key[-1] == "shuffled" else False

            # Single session decoding
            if ex_key[0] == "single":

                # Only pass sessions with more units than the target # of components
                valid_sessions = []
                for session in self.sessions:
                    X = session.filter_X(unit_types=unit_types)
                    C_in = X.shape[-1]
                    if self.kwargs["n_components"] is not None and C_in < self.kwargs["n_components"]:
                        valid_sessions.append(False)
                    else:
                        valid_sessions.append(True)

                # Run experiments
                for session, flag in zip(self.sessions, valid_sessions):
                    print(f"Working on experiment {i_job + 1} out of {n_jobs}")
                    if flag:
                        ex = SingleDecodingExperiment(
                            session,
                            shuffle_trials=shuffle_trials,
                            **self.kwargs
                        )
                        ex.run(unit_types=unit_types)
                        self.experiments[ex_key].append(ex)
                    i_job += 1

            # Pseudo-session decoding
            else:
                for i_run in range(self.n_runs):
                    print(f"Working on experiment {i_job + 1} out of {n_jobs}")
                    pseudosession.reseed(i_run)
                    ex = SingleDecodingExperiment(
                        pseudosession,
                        shuffle_trials=shuffle_trials,
                        **self.kwargs
                    )
                    ex.run(unit_types=unit_types)
                    self.experiments[ex_key].append(ex)
                    i_job += 1

        print("All done!")

        return

    def visualize(self, figsize=(8, 4)):
        """
        """

        fig, axs = plt.subplots(nrows=2, ncols=5, sharex=True)

        #
        t = self.experiments[("single", "visual", "unshuffled")][0].t
        ks = (
            "saccade_direction",
            "saccade_amplitude",
            "saccade_startpoints",
            "saccade_endpoints",
            "saccade_velocity"
        )

        # Plot curves for single sessions
        for j, k in enumerate(ks):

            # Unshuffled
            for u, c in zip(["visual", "premotor"], ["green", "purple"]):
                ys = list()
                exs = self.experiments[("single", u, "unshuffled")]
                for ex in exs:
                    mask = ex.result.feature == k
                    steps = ex.result.step[mask]
                    scores = ex.result.score[mask]
                    idx = np.argsort(steps)
                    steps = steps[idx]
                    scores = scores[idx]
                    y = np.full_like(t, np.nan, dtype=float)
                    y[steps] = scores
                    ys.append(y)
                ys = np.vstack(ys)
                y_mean = np.nanmean(ys, axis=0)
                y_std = np.nanstd(ys, axis=0)
                axs[0, j].plot(t, y_mean, color=c, label=u)
                axs[0, j].fill_between(t, y_mean - y_std, y_mean + y_std, color=c, alpha=0.25, edgecolor="none")

            # Shuffled
            for u in ["visual", "premotor"]:
                ys = list()
                exs = self.experiments[("single", u, "shuffled")]
                for ex in exs:
                    mask = ex.result.feature == k
                    steps = ex.result.step[mask]
                    scores = ex.result.score[mask]
                    idx = np.argsort(steps)
                    steps = steps[idx]
                    scores = scores[idx]
                    y = np.full_like(t, np.nan, dtype=float)
                    y[steps] = scores
                    ys.append(y)
                ys = np.vstack(ys)
                y_mean = np.nanmean(ys, axis=0)
                y_std = np.nanstd(ys, axis=0)
                axs[0, j].plot(t, y_mean, color="0.5", label=u)
                axs[0, j].fill_between(t, y_mean - y_std, y_mean + y_std, color="0.5", alpha=0.25, edgecolor="none")

        # Plot curves for pseudosessions
        for j, k in enumerate(ks):

            # Unshuffled
            for u, c in zip(["visual", "premotor"], ["green", "purple"]):
                ys = list()
                exs = self.experiments[("pseudo", u, "unshuffled")]
                for ex in exs:
                    mask = ex.result.feature == k
                    steps = ex.result.step[mask]
                    scores = ex.result.score[mask]
                    idx = np.argsort(steps)
                    steps = steps[idx]
                    scores = scores[idx]
                    y = np.full_like(t, np.nan, dtype=float)
                    y[steps] = scores
                    ys.append(y)
                ys = np.vstack(ys)
                y_mean = np.nanmean(ys, axis=0)
                y_std = np.nanstd(ys, axis=0)
                axs[1, j].plot(t, y_mean, color=c, label=u)
                axs[1, j].fill_between(t, y_mean - y_std, y_mean + y_std, color=c, alpha=0.25, edgecolor="none")

            # Shuffled
            for u in ["visual", "premotor"]:
                ys = list()
                exs = self.experiments[("pseudo", u, "shuffled")]
                for ex in exs:
                    mask = ex.result.feature == k
                    steps = ex.result.step[mask]
                    scores = ex.result.score[mask]
                    idx = np.argsort(steps)
                    steps = steps[idx]
                    scores = scores[idx]
                    y = np.full_like(t, np.nan, dtype=float)
                    y[steps] = scores
                    ys.append(y)
                ys = np.vstack(ys)
                y_mean = np.nanmean(ys, axis=0)
                y_std = np.nanstd(ys, axis=0)
                axs[1, j].plot(t, y_mean, color="0.5", label=u)
                axs[1, j].fill_between(t, y_mean - y_std, y_mean + y_std, color="0.5", alpha=0.25, edgecolor="none")

        #
        ylim = [np.inf, -np.inf]
        for ax in axs.flatten():
            y1, y2 = ax.get_ylim()
            ylim[0] = min(ylim[0], y1)
            ylim[1] = max(ylim[1], y2)
        y1, y2 = ylim
        for ax in axs.flatten():
            ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])

        #
        fig.supylabel(r"$R^2$ (cross-validated)", fontsize=10)
        titles = ["Direction", "Amplitude", "Startpoint", "Endpoint", "Velocity"]
        for ax, title in zip(axs[0, :], titles):
            ax.set_title(title, fontsize=10)
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])
        fig.supxlabel("Time from saccade initiation (s)", fontsize=10)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs  
