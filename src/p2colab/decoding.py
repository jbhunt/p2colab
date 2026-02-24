import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .model_interfaces import Decoder

class NeuralActivityProcessor():
    """
    """

    def __init__(self, n_components=3):
        """
        """

        self.n_components= n_components

        return
    
    def fit(self, X):
        """
        """

        N, T, C = X.shape
        X_new = X.reshape(N * T, C)
        self.tf1 = StandardScaler().fit(X_new)
        self.tf2 = PCA(n_components=self.n_components)
        self.tf2.fit(self.tf1.transform(X_new))

        return
    
    def fit_transform(self, X):
        """
        """

        self.fit(X)
        out = self.transform(X)

        return out
    
    def transform(self, X):
        """
        """

        N, T, C = X.shape
        X_new = X.reshape(N * T, C)
        out = self.tf1.transform(X_new)
        out = self.tf2.transform(out)
        out = out.reshape(N, T, self.n_components)

        return out

class Result():
    """
    """

    _score = None
    _k = None
    _step = None

    def __init__(self, score=None, feature=None, split=None, j=None, step=None):
        """
        """

        self._score = score
        self._feature = feature
        self._j = j
        self._split = split
        self._step = step

        return
    
    def merge_with(self, r):
        """
        """

        attrs = ("_score", "_feature", "_split", "_j", "_step")
        for attr in attrs:
            a1 = getattr(self, attr)
            if a1 is None:
                a1 = np.array([])
            else:
                a1 = np.atleast_1d(a1)
            a2 = getattr(r, attr)
            if a2 is None:
                a2 = np.array([])
            else:
                a2 = np.atleast_1d(a2)
            out = np.concatenate([a1, a2])
            setattr(self, attr, out)

        return

    @property
    def score(self):
        return self._score
    
    @property
    def j(self):
        return self._j
    
    @property
    def feature(self):
        return self._feature

    @property
    def split(self):
        return self._split
    
    @property
    def step(self):
        return self._step
    
class SingleDecodingExpeirment():
    """
    """

    def __init__(
        self,
        ds,
        window_size=30,
        stride=5,
        n_components=3,
        train_size=0.7,
        validation_size=0.1,
        kernel_size=5,
        lr=0.001,
        max_iter=100,
        batch_size=None
        ):
        """
        """

        self.ds = ds
        self.train_size = train_size
        self.validation_size = validation_size
        self.est = Decoder(
            n_components,
            1,
            kernel_size=kernel_size,
            lr=lr,
            max_iter=max_iter,
            batch_size=batch_size
        )
        self.window_size = window_size
        self.stride = stride
        self.result = None
        self.n_components = n_components

        # Set time (for plotting)
        N, T, C = self.ds.X.shape
        left_edges = np.arange(0, T - self.window_size + 1, self.stride)
        right_edges = left_edges + self.window_size
        all_edges = np.hstack([left_edges.reshape(-1, 1), right_edges.reshape(-1, 1)])
        self.t = self.ds.t_X[right_edges - 1]

        return
    
    def run(self, unit_types=["premotor", "visuomotor", "visual"], split_seeds=[0, 1, 2]):
        """
        """

        N, T, C = self.ds.X.shape
        starts = np.arange(0, T - self.window_size + 1, self.stride)
        ends = starts + self.window_size
        window_edges = np.column_stack((starts, ends))
        n_steps = window_edges.shape[0]
        ks = (
            "saccade_direction",
            "saccade_amplitude",
            "saccade_startpoints",
            "saccade_endpoints"
        )
        self.result = Result()
        n_splits = len(split_seeds)
        n_jobs = n_splits * n_steps * 4
        i_job = 0

        # For each train, validation, test split ...
        for i_split in range(n_splits):

            # Make splits
            ds_train, ds_valid, ds_test = self.ds.random_split(
                [self.train_size, self.validation_size, 1 - sum([self.train_size, self.validation_size])],
                split_seed=split_seeds[i_split]
            )

            # Standardize and decompose neural data
            X_tf = NeuralActivityProcessor(n_components=self.n_components)
            X_train = ds_train.filter_X(unit_types=unit_types)
            X_train = X_tf.fit_transform(X_train)
            X_valid = ds_valid.filter_X(unit_types=unit_types)
            X_valid = X_tf.transform(X_valid)
            X_test = ds_test.filter_X(unit_types=unit_types)
            X_test = X_tf.transform(X_test)

            # For each kinematic feature
            for j, k in enumerate(ks):

                # Standardize target kinematic feature
                y_tf = StandardScaler()
                y_tf.fit(getattr(ds_train, k).reshape(-1, 1))
                for ds in [ds_train, ds_valid, ds_test]:
                    y = getattr(ds, k).reshape(-1, 1)
                    y = y_tf.transform(y).flatten()
                    ds.set_y(y)

                # Move through time
                for i_step, (start, stop) in enumerate(window_edges):

                    # Slice out window of neural data
                    for ds, X in zip([ds_train, ds_valid, ds_test], [X_train, X_valid, X_test]):
                        ds.set_X(X[:, start: stop, :])

                    #
                    end = "\r" if (i_job + 1) < n_jobs else "\n"
                    message = f"Working on job {i_job + 1} out of {n_jobs}"
                    print(message, end=end)

                    # Fit
                    self.est._return_to_initial_state()
                    self.est.fit(ds_train, ds_valid, print_info=False)

                    # Evaluate performance
                    y_true = ds_test.y
                    y_pred = self.est.predict(ds_test).flatten()
                    r2 = 1 - (np.sum(np.power(y_true - y_pred, 2)) / np.sum(np.power(y_true - y_true.mean(), 2)))
                    # rmse = round(np.sqrt(np.mean(np.power(y_true.flatten() - y_pred.flatten(), 2))).item(), 6)

                    #
                    result = Result(score=r2, feature=k, split=i_split, j=j, step=i_step)
                    self.result.merge_with(result)

                    # Reset datasets
                    for ds in (ds_train, ds_valid, ds_test):
                        ds.reset_X()

                    #
                    i_job += 1

                #
                for ds in (ds_train, ds_valid, ds_test):
                    ds.reset_y()

        return
    
class AllDecodingExperiments():
    """
    """

    def __init__(self, sessions):
        """
        """

        self.sessions = sessions
        self.experiments = None

        return
    
    def run(self, n_components=3, stride=1, split_seeds=[0, 1, 2]):
        """
        """

        #
        self.experiments = {
            # Single session
            "s": {
                "vo": list(),
                "vm": list(),
                "pm": list(),
                "nf": list()
            },
            # Multi-session
            "m": {
                "vo": list(),
                "vm": list(),
                "pm": list(),
                "nf": list()
            },
        }
        keys = ("vo", "vm", "pm","nf")
        sets = (
            ["visual"],
            ["visuomotor"],
            ["premotor"],
            ["visual", "visuomotor", "premotor"],
            # None
        )
        for k, s in zip(keys, sets):

            sessions = list()
            for ds in self.sessions:
                X = ds.filter_X(unit_types=s)
                C_in = X.shape[-1]
                if C_in < n_components:
                    continue
                sessions.append(ds)

            #
            for ds in sessions:
                ex = SingleDecodingExpeirment(ds, n_components=n_components, stride=stride)
                ex.run(unit_types=s, split_seeds=split_seeds)
                self.experiments["s"][k].append(ex)

        return
    
    def visualize(self, figsize=(8, 5)):
        """
        """

        fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True)
        t = self.experiments["s"]["vo"][0].t # Timestamp of the bin on the right edge of the window
        ks = (
            "saccade_direction",
            "saccade_amplitude",
            "saccade_startpoints",
            "saccade_endpoints"
        )
        for (j, s, c) in zip([0, 1, 2, 3], ["vo", "vm", "pm", "nf"], ["C0", "C1", "C2", "C3"]):

            #
            for i, k in enumerate(ks):
            
                #
                ys = list()
                for ex in self.experiments["s"][s]:

                    #
                    mask = (ex.result.feature == k)
                    scores = ex.result.score[mask]
                    splits = ex.result.split[mask]
                    n_splits = len(np.unique(splits))
                    scores = scores.reshape(n_splits, -1)
                    y = scores.mean(0)
                    axs[i, j].plot(t, y, color=c, lw=1.0, alpha=0.3)
                    ys.append(y)
                
                #
                ys = np.array(ys)
                axs[i, j].plot(t, ys.mean(0), color=c)

        #
        ylim = [np.inf, -np.inf]
        for ax in axs.flatten():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        y1, y2 = ylim
        for ax in axs.flatten():
            ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])

        #
        fig.supylabel(r"$R^2$ (cross-validated)", fontsize=10)
        ylabels = ["Direction","Amplitude", "Startpoint", "Endpoint"]
        for ax, ylabel in zip(axs[:, 0], ylabels):
            ax.set_ylabel(ylabel)
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])
        fig.supxlabel("Time from saccade initiation", fontsize=10)
        titles = [
            "Visual-only", "Visuomotor", "Premotor", "All units",
        ]
        for ax, t in zip(axs[0, :].flatten(), titles):
            ax.set_title(t, fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs        
