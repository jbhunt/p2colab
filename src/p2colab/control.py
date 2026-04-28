import numpy as np
from matplotlib import pyplot as plt
from .encoding import SinglePermutationExperiment
from .decoding import SingleDecodingExperiment
from .datasets import DummyMlatiDataset
from .utils import NeuralActivityProcessor
from scipy.stats import rankdata
import matplotlib.colors as mc

def match_empirical_distribution(y, x):
    """
    """

    y = np.asarray(y).reshape(-1)
    x = np.asarray(x).reshape(-1)
    x_sorted = np.sort(x)
    ranks = rankdata(y, method="ordinal") - 1

    return x_sorted[ranks]

def standardize(x):
    """
    """

    x = np.asarray(x).reshape(-1)

    return (x - x.mean()) / x.std()

def generate_nuisance_regressor(y, r, rng):
    """
    """

    #
    y = standardize(y)
    e = rng.normal(size=y.shape[0])
    e = standardize(e)

    # Remove sample correlation between e and y
    e = e - (np.dot(e, y) / np.dot(y, y)) * y
    e = standardize(e)

    #
    z = r * y + np.sqrt(1 - r ** 2) * e
    z = standardize(z)

    return z

def generate_Xy(ds, unit_types, dropout, reference_feature):
    """
    """

    #
    rng = np.random.default_rng()
    est = NeuralActivityProcessor(n_components=None)

    # Norm neural activity
    X_raw = ds.filter_X(unit_types)
    N, T, C = X_raw.shape
    X_norm = est.fit_transform(X_raw)
    X = X_norm.reshape(N, T * C)

    # Generate time-varying weights with dropout
    W = np.zeros([T, C])
    time_idx = np.where(np.logical_and(ds.t_X < 0, ds.t_X >= -0.1))[0]
    W[time_idx, :] = 1.0
    if dropout != 0:
        unit_idx = rng.choice(np.arange(C), size=round(C * dropout))
        W[:, unit_idx] = 0.0
    W = W.ravel()

    # Build neural-derived latent
    y = X @ W
    y = standardize(y)
    y = y + rng.normal(loc=0, scale=0.5, size=N)

    # Match target distribution, then standardize again
    x_real = np.asarray(getattr(ds, reference_feature)).reshape(-1)
    y = match_empirical_distribution(y, x_real)
    y = standardize(y)

    return X, y

class EncodingControlExperiment(SinglePermutationExperiment):
    """
    """

    def __init__(
        self,
        ds,
        max_regs=5,
        rs=np.linspace(0, 1, 11),
        dropout=0.1,
        reference_feature="saccade_amplitude",
        alpha=0.05,
        signal_center=-0.05,
        signal_sigma=0.07,
        signal_amplitude=1.0,
        **kwargs
        ):
        """
        """

        kwargs_ = {
            "n_components": 3
        }
        kwargs_.update(kwargs)

        self.ds = ds
        self.max_regs = max_regs
        self.rs = rs
        self.dropout = dropout
        self.reference_feature = reference_feature
        self.alpha = alpha

        # Parameters controlling the synthetic temporal signal
        self.signal_center = signal_center
        self.signal_sigma = signal_sigma
        self.signal_amplitude = signal_amplitude

        # kwargs forwarded to SinglePermutationExperiment
        self.kwargs = kwargs_

        # Results
        self.result = None
        self.result_stat = None
        self.result_sig = None
        self.experiments = None
        self.signal_gain = None
        self.signal_index = None

        return
    
    def run(self, unit_types=["premotor", "visuomotor"]):
        """
        """

        rng = np.random.default_rng()
        est = NeuralActivityProcessor(n_components=None)
        jobs = []

        # Real neural activity
        X_raw = self.ds.filter_X(unit_types)
        N, T, C = X_raw.shape

        # Normalize neural activity
        X_norm = est.fit_transform(X_raw)
        X_reshaped = X_norm.reshape(N, T * C)

        # Generate smooth time-varying weights with dropout
        t = self.ds.t_X
        gain = self.signal_amplitude * np.exp(-0.5 * ((t - self.signal_center) / self.signal_sigma) ** 2)
        W = np.tile(gain[:, None], (1, C))                              # (T, C)

        if self.dropout != 0:
            unit_idx = rng.choice(
                np.arange(C),
                size=round(C * self.dropout),
                replace=False
            )
            W[:, unit_idx] = 0.0

        W = W.ravel()

        # Build neural-derived latent target regressor
        y = X_reshaped @ W
        y = standardize(y)
        y = y + rng.normal(loc=0, scale=0.5, size=N)

        # Match target distribution, then standardize again
        x_real = np.asarray(getattr(self.ds, self.reference_feature)).reshape(-1)
        y = match_empirical_distribution(y, x_real)
        y = standardize(y)

        # Build design matrices
        for r in self.rs:
            all_regs = []
            for _ in range(self.max_regs):
                z = generate_nuisance_regressor(y, r, rng)
                all_regs.append(z)

            for n_regs in range(1, self.max_regs + 1):
                X_design = np.column_stack([y] + all_regs[:n_regs])
                jobs.append((r, n_regs, X_design))

        # Evaluate at the time bin closest to the signal center
        i_t = np.argmin(np.abs(self.ds.t_X - self.signal_center))

        n_rows = self.max_regs
        n_cols = len(self.rs)
        n_jobs = n_rows * n_cols
        self.result = {
            "partial_r2": np.full((2, n_rows, n_cols), np.nan),
            "pvalues": np.full((2, n_rows, n_cols), np.nan)
        }
        self.experiments = []

        for i_job, (r, n, X_design) in enumerate(jobs):
            print(f"Working on job {i_job + 1} out of {n_jobs}")

            ks = np.concatenate([
                np.asarray(["signal_0"]),
                np.asarray([f"nuisance_{i}" for i in range(n)])
            ])

            ex = SinglePermutationExperiment(self.ds, ks=ks, alpha=self.alpha, **self.kwargs)
            ex.run(unit_types=unit_types, X_norm=X_design)
            self.experiments.append({
                "r": r,
                "n_regs": n,
                "ex": ex,
            })
            for (h, name) in zip([0, 1], ["signal_0", "nuisance_0"]):
                mask = ex.result.feature == name
                steps = ex.result.time[mask]
                stats = ex.result.statistic[mask]
                step_idx = np.argsort(steps)
                stats = stats[step_idx]
                r2 = stats[i_t]
                unique_features = np.unique(ex.result.feature)
                feat_idx = np.where(unique_features == name)[0].item()
                pvalues = np.asarray(ex.result.pvalue[feat_idx])[step_idx]
                i = n - 1
                j = np.where(np.asarray(self.rs) == r)[0].item()
                self.result["partial_r2"][h, i, j] = r2
                self.result["pvalues"][h, i, j]= pvalues[i_t]

        return

    def visualize(self, alpha=0.05, v_range=(0, 0.5), figsize=(7, 2.7)):
        """
        """

        fig, axs = plt.subplots(
            ncols=2,
            nrows=2,
            sharey=True,
            constrained_layout=True
        )

        if v_range is None:
            vmin = 0
            vmax = 0.3
        else:
            vmin, vmax = v_range

        y = np.arange(self.max_regs) + 1
        x = np.arange(len(self.rs))

        #
        bool_cm = mc.ListedColormap(["black", "white"])
        for h in (0, 1):
            partial_r2 = self.result["partial_r2"][h]
            im_r2 = axs[0, h].pcolor(x, y, partial_r2, vmin=vmin, vmax=vmax)
            pvalues = self.result["pvalues"][h]
            im_p = axs[1, h].pcolor(x, y, pvalues < alpha, cmap=bool_cm, vmin=0, vmax=1)

        #
        xticks = np.interp([0, 0.5, 1.0], [0, 1.0], [0, x.size - 1])
        yticks = np.array([1.0, 5.0, 10.0, 15.0])
        for ax in axs.flatten():
            ax.set_xticks(xticks)
            ax.set_xticklabels([0, 0.5, 1.0])
            ax.set_yticks(yticks)
        
        #
        cb_r2 = fig.colorbar(im_r2, ax=axs[0, -1], shrink=0.8, aspect=10)
        cb_r2.set_label(r"Partial $R^2$")
        cb_p = fig.colorbar(im_p, ax=axs[1, -1], shrink=0.8, aspect=10)
        cb_p.set_label(f"p < {alpha:.2f}")
        cb_p.set_ticks([0.25, 0.75])
        cb_p.set_ticklabels(["No", "Yes"])
        fig.supxlabel("Corr.", fontsize=10)
        fig.supylabel("# of nuisance regressors", fontsize=10)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

        return fig, axs
    
class DecodingControlExperiment():
    """
    Shows decoding of a nuisance regressor as a function of time and correlation
    """

    def __init__(
        self,
        reference_session,
        reference_feature="saccade_amplitude",
        rs=np.linspace(0, 1, 11),
        dropout=0.1,
        **kwargs
        ):
        """
        """

        kwargs_ = {
            "n_components": 3,
            "ks": ("saccade_amplitude", "saccade_startpoints")
        }
        kwargs_.update(kwargs)
        self.kwargs = kwargs_
        self.rs = rs
        self.dropout = dropout
        self.reference_session = reference_session
        self.reference_feature = reference_feature

        return
    
    def run(self, unit_types=["premotor", "visuomotor"]):
        """
        """

        rng = np.random.default_rng()

        # X_real - Real neural activity
        # X_syn  - Synthetic neural activity
        # y_targ - Target kinematic feature
        # y_nuis - Nuisance kinematic feature
        X_real = getattr(self.reference_session, "X") # shape: (N, K) or (N,)
        N, T, C = X_real.shape
        y_targ = getattr(self.reference_session, self.reference_feature)
        if y_targ.ndim == 1:
            y_targ = y_targ[:, None]
        # W = np.full([1, C], 1.0)
        W = np.random.normal(size=[1, C], loc=0, scale=1)
        idx = np.random.choice(np.arange(C), size=int(self.dropout * C))
        W[:, idx] = 0.0
        t = self.reference_session.t_X
        center = -0.05 # peak at -50 ms
        sigma = 0.07 # 70 ms width
        gain = 0.8 * np.exp(-0.5 * ((t - center) / sigma) ** 2)
        X_syn = y_targ @ W
        X_syn = X_syn[:, None, :] * gain[None, :, None]     # (N, T, C)
        noise_level = 5.0
        X_syn += rng.normal(scale=noise_level, size=(N, T, C))

        #
        sessions = list()
        for r in self.rs:
            y_nuis = generate_nuisance_regressor(y_targ, r, rng)
            ys = np.column_stack([y_targ, y_nuis])
            y_labels = ["saccade_amplitude", "saccade_startpoints"]
            s = DummyMlatiDataset(self.reference_session, X_syn, ys, y_labels, default_target="saccade_amplitude")
            sessions.append(s)

        #
        experiments = list()
        for s in sessions:
            ex = SingleDecodingExperiment(s, **self.kwargs)
            ex.run(unit_types=unit_types)
            experiments.append(ex)

        #
        n_t = self.reference_session.t_X.size
        n_r = len(self.rs)
        self.result = np.full([2, n_r, n_t], np.nan) # (2, N correlation levels, N time points) 
        for z, k in enumerate(["saccade_amplitude", "saccade_startpoints"]):
            for i_r in range(n_r):
                ex = experiments[i_r]
                mask = ex.result.feature == k
                r2_values = ex.result.score[mask]
                steps = ex.result.step[mask]
                idx = np.argsort(steps)
                r2_values = r2_values[idx]
                self.result[z, i_r, :] = r2_values

        #
        return
    
    def visualize(self, figsize=(7, 1.7)):
        """
        """

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, constrained_layout=True)
        t = self.reference_session.t_X
        im = ax1.pcolor(t, self.rs, self.result[0], vmin=0, vmax=1)
        ax2.pcolor(t, self.rs, self.result[1], vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=(ax1, ax2), shrink=0.9, aspect=10)
        cb.set_label(r"$R^2$")
        ax1.set_title("Signal regressor", fontsize=10)
        ax2.set_title("Nuisance regressor", fontsize=10)
        fig.supxlabel("Time form saccade initiation (s)", fontsize=10)
        fig.supylabel("Corr.", fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

        return fig, (ax1, ax2)