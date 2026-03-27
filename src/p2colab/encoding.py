import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from .datasets import PsuedoSessionDataset
from .utils import NeuralActivityProcessor

# Psuedocode
# ----------
# For each kinematic variable ...
# For each time bin ...
# Partition data into X (target variable) and Z (nuisance variables)
# Fit full model (X + Z) and reduced model (Z only)
# Compute error and RSS for the full (1) and reduced (0) models
# Compute partial R^2 ([RSS_0 - RSS_1] / RSS_0) (This is the test statistic)
# For each permuation ...
# Permute residuals from the reduced model (E_0)
# Reconstruct target by adding shuffled residuals to reduced model predictions (this is where the magic happens)
# Refit reduced and full models using augmented target
# Recompute partial R^2 for the reduced and full models
# Generate null distribution of the partial R^2 statistic
# Compute p-value(s) (correct for many comparisons across features and time)

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import false_discovery_control
import time

class Result:
    """
    """

    def __init__(self):
        self._feature = []
        self._time = []
        self._statistic = []
        self._null = []
        self._r2 = []
        self._f = []
        self._pvalue = None

    def cache(self, feature, time, statistic, null):
        """
        Store one observed statistic and its permutation null distribution
        """

        null = np.asarray(null)

        if null.ndim != 1:
            raise ValueError("null must be a 1D array of permutation statistics")

        self._feature.append(feature)
        self._time.append(time)
        self._statistic.append(statistic)
        self._r2.append(statistic)
        self._f.append(statistic)
        self._null.append(null)

    def process(self):
        """
        """

        self._feature = np.asarray(self._feature)
        self._time = np.asarray(self._time)
        self._statistic = np.asarray(self._statistic)
        self._null = np.asarray(self._null)
        self._r2 = np.asanyarray(self._r2)
        self._f = np.asarray(self._r2)
        self.compute_pvalues()

        return

    def reshape_statistics(self):
        """
        """

        feature_vals = np.unique(self.feature)
        time_vals = np.unique(self.time)

        out = np.full((len(feature_vals), len(time_vals)), np.nan)

        for i, f in enumerate(feature_vals):
            for j, t in enumerate(time_vals):
                m = (self.feature == f) & (self.time == t)
                if m.sum() != 1:
                    raise ValueError(
                        f"Expected exactly one statistic for feature={f}, time={t}, found {m.sum()}"
                    )
                out[i, j] = self.statistic[m][0]

        return out

    def reshape_nulls(self):
        """
        """

        feature_vals = np.unique(self.feature)
        time_vals = np.unique(self.time)

        if len(self.null) == 0:
            raise ValueError("No null distributions have been cached")

        n_perm = self.null.shape[1]
        out = np.full((len(feature_vals), len(time_vals), n_perm), np.nan)

        for i, f in enumerate(feature_vals):
            for j, t in enumerate(time_vals):
                m = (self.feature == f) & (self.time == t)
                if m.sum() != 1:
                    raise ValueError(
                        f"Expected exactly one null for feature={f}, time={t}, found {m.sum()}"
                    )
                out[i, j, :] = self.null[m][0]

        return out

    def compute_pvalues(self, adjustment="fwer", family="time"):
        """
        """

        stats = self.reshape_statistics()      # (J, T)
        nulls = self.reshape_nulls()           # (J, T, P)
        _, _, n_perm = nulls.shape

        ge = nulls >= stats[:, :, None]
        p_raw = (1 + ge.sum(axis=2)) / (1 + n_perm)
        
        #
        if adjustment in [None, False]:
            p_adjusted = p_raw

        elif adjustment == "fdr":
            if family == "all":
                p_adjusted = false_discovery_control(
                    p_raw.ravel(),
                    method="bh"
                ).reshape(p_raw.shape)

            elif family == "time":
                p_adjusted = np.empty_like(p_raw)
                for j in range(p_raw.shape[0]):
                    p_adjusted[j] = false_discovery_control(
                        p_raw[j],
                        method="bh"
                    )

            else:
                raise ValueError("family must be 'all' or 'time'")

        elif adjustment == "fwer":
            if family == "all":
                max_null = nulls.max(axis=(0, 1))            # (P,)
                ge = max_null[None, None, :] >= stats[:, :, None]
                p_adjusted = (1 + ge.sum(axis=2)) / (1 + n_perm)

            elif family == "time":
                max_null = nulls.max(axis=1)                 # (J, P)
                ge = max_null[:, None, :] >= stats[:, :, None]
                p_adjusted = (1 + ge.sum(axis=2)) / (1 + n_perm)

            else:
                raise ValueError("family must be 'all' or 'time'")

        else:
            raise ValueError("adjustment must be None, False, 'fdr', or 'fwer'")

        self._pvalue = p_adjusted
        return p_adjusted

    @property
    def feature(self):
        return self._feature

    @property
    def time(self):
        return self._time

    @property
    def statistic(self):
        return self._statistic
    
    @property
    def f(self):
        return self._f
    
    @property
    def r2(self):
        return self._r2

    @property
    def null(self):
        return self._null

    @property
    def pvalue(self):
        return self._pvalue

class SinglePermutationExperiment():
    """
    """

    ks = (
        "saccade_direction",
        "saccade_amplitude",
        "saccade_startpoints",
        "saccade_endpoints",
        "saccade_velocity",
    )

    def __init__(self, ds, n_permutations=100, n_components=3, alpha=0.0):
        """
        """

        self.ds = ds
        self.result = None
        self.n_permutations = n_permutations
        self.n_components = n_components
        self.alpha = alpha

        return

    def run(self, unit_types=["visual", "premotor", "visuomotor"]):
        """
        """

        # Build baseline design matrix
        X = np.vstack([
            self.ds.saccade_direction,
            self.ds.saccade_amplitude,
            self.ds.saccade_startpoints,
            self.ds.saccade_endpoints,
            self.ds.saccade_velocity
        ]).T
        X_norm = StandardScaler().fit_transform(X)

        # Transform neural target
        y_raw = self.ds.filter_X(unit_types)
        proc = NeuralActivityProcessor(n_components=self.n_components)
        y_norm = proc.fit_transform(y_raw)
        N, T, _ = y_norm.shape

        # Build within-block permutations
        unique_blocks = np.unique(self.ds.saccade_blocks)
        block_indices = {
            b: np.where(self.ds.saccade_blocks == b)[0]
            for b in unique_blocks
        }
        perms = np.empty((self.n_permutations, N), dtype=int)
        for p in range(self.n_permutations):
            perm = np.empty(N, dtype=int)
            for _, idx in block_indices.items():
                perm[idx] = np.random.permutation(idx)
            perms[p] = perm

        # Initialize model
        if self.alpha == 0:
            model = LinearRegression(fit_intercept=True)
        else:
            model = Ridge(alpha=self.alpha, fit_intercept=True)

        # Initialize result object
        self.result = Result()
        n_features = X_norm.shape[1]
        n_jobs = n_features * T * self.n_permutations
        i_job = 0

        # For each kinematic variable
        for j, k in enumerate(self.ks):

            # For each time bin
            for t in range(T):

                y_t = y_norm[:, t, :]   # shape: (N, C_components)

                # Reduced and full models
                X_0 = np.delete(X_norm, j, axis=1)
                X_1 = X_norm

                # Reduced fit
                model.fit(X_0, y_t)
                y_0 = model.predict(X_0).reshape(N, -1)
                E_0 = y_t - y_0
                rss_0 = np.sum(E_0 ** 2)

                # Full fit
                model.fit(X_1, y_t)
                E_1 = y_t - model.predict(X_1).reshape(N, -1)
                rss_1 = np.sum(E_1 ** 2)

                # Observed partial R^2
                statistic = (rss_0 - rss_1) / rss_0

                # Null distribution
                null = np.full(self.n_permutations, np.nan)

                for p in range(self.n_permutations):
                    end = "\r" if (i_job + 1) < n_jobs else "\n"
                    print(f"Running fit {i_job + 1} out of {n_jobs}", end=end)

                    index = perms[p]
                    E_0_shuffled = E_0[index, :]
                    y_null = y_0 + E_0_shuffled

                    model.fit(X_0, y_null)
                    E_0_null = y_null - model.predict(X_0).reshape(N, -1)
                    rss_0_null = np.sum(E_0_null ** 2)

                    model.fit(X_1, y_null)
                    E_1_null = y_null - model.predict(X_1).reshape(N, -1)
                    rss_1_null = np.sum(E_1_null ** 2)

                    # null[p] = (rss_0 - rss_1_null) / rss_0
                    null[p] = (rss_0_null - rss_1_null) / rss_0_null

                    i_job += 1

                self.result.cache(
                    feature=k,
                    time=t,
                    statistic=statistic,
                    null=null,
                )

        #
        self.result.process()

        return
    
class AllPermutationExperiments():
    """
    """

    def __init__(self, sessions, n_runs=10, **kwargs):
        """
        """

        self.sessions = sessions
        self.experiments = None
        self.n_runs = n_runs
        self.kwargs = {
            "n_components": 3,
            "n_permutations": 100,
            "alpha": 0.0,
        }
        self.kwargs.update(kwargs)

        return
    
    def run(self):
        """
        """
    
        pseudosession = PsuedoSessionDataset(
            sessions=self.sessions,
            build=False
        )
        self.experiments = {
            "single": {  # single session
                "visual": [],
                "premotor": [],
            },
            "pseudo": {  # multi-session (not yet used here)
                "visual": [],
                "premotor": [],
            },
        }
        keys = ("visual", "premotor")
        sets = (
            ["visual"],
            ["visuomotor", "premotor"],
        )
        n_jobs = (self.n_runs * 2) + (len(self.sessions) * 2)
        i_job = 0
        for k, s in zip(keys, sets):

            # Only pass sessions with more units than the target # of components
            valid_sessions = []
            for session in self.sessions:
                X = session.filter_X(unit_types=s)
                C_in = X.shape[-1]
                if self.kwargs["n_components"] is not None and C_in < self.kwargs["n_components"]:
                    valid_sessions.append(False)
                else:
                    valid_sessions.append(True)

            # Run the experiments
            for session, flag in zip(self.sessions, valid_sessions):
                print(f"Working on experiment {i_job + 1} out of {n_jobs}")
                if flag:
                    ex = SinglePermutationExperiment(
                        session,
                        **self.kwargs
                    )
                    ex.run(unit_types=s)
                    self.experiments["single"][k].append(ex)
                i_job += 1

            #
            for i_run in range(self.n_runs):
                pseudosession.reseed(i_run)
                ex = SinglePermutationExperiment(
                    pseudosession,
                    **self.kwargs
                )
                print(f"Working on experiment {i_job + 1} out of {n_jobs}")
                ex.run(unit_types=s)
                self.experiments["pseudo"][k].append(ex)
                i_job += 1

        print("All done!")

        return
    
    def visualize(self, figsize=(10, 6), alpha=0.05):
        """
        """

        fig, axs = plt.subplots(
            nrows=4,
            ncols=5,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 3, 1]}
        )

        ks = (
            "saccade_direction",
            "saccade_amplitude",
            "saccade_startpoints",
            "saccade_endpoints",
            "saccade_velocity",
        )
        titles = ["Direction", "Amplitude", "Startpoint", "Endpoint", "Velocity"]

        # Use time axis from first available experiment
        t = self.experiments["single"]["visual"][0].ds.t_X

        # Single sessions
        for j, k in enumerate(ks):
            for u, c in zip(["visual", "premotor"], ["green", "purple"]):
                ys = []
                ps = []

                for ex in self.experiments["single"][u]:
                    feature = np.asarray(ex.result.feature)
                    time = np.asarray(ex.result.time)
                    statistic = np.asarray(ex.result.statistic)

                    if ex.result.pvalue is None:
                        pvalue_mat = ex.result.compute_pvalues()
                    else:
                        pvalue_mat = ex.result.pvalue

                    unique_features = np.unique(feature)
                    unique_times = np.unique(time)

                    if k not in unique_features:
                        continue

                    i_feature = np.where(unique_features == k)[0][0]

                    y = np.full(len(t), np.nan, dtype=float)
                    p = np.full(len(t), np.nan, dtype=float)

                    mask = feature == k
                    time_k = time[mask]
                    stat_k = statistic[mask]

                    idx = np.argsort(time_k)
                    time_k = time_k[idx]
                    stat_k = stat_k[idx]

                    # Fill statistic curve
                    if np.issubdtype(np.asarray(time_k).dtype, np.integer):
                        valid = (time_k >= 0) & (time_k < len(t))
                        y[time_k[valid]] = stat_k[valid]

                        # Fill p-values using the same integer indexing
                        for tt in time_k[valid]:
                            t_idx = np.where(unique_times == tt)[0][0]
                            p[tt] = pvalue_mat[i_feature, t_idx]
                    else:
                        time_to_index = {tt: i for i, tt in enumerate(t)}
                        for tt, ss in zip(time_k, stat_k):
                            if tt in time_to_index:
                                i_t = time_to_index[tt]
                                y[i_t] = ss
                                t_idx = np.where(unique_times == tt)[0][0]
                                p[i_t] = pvalue_mat[i_feature, t_idx]

                    ys.append(y)
                    ps.append(p)

                if len(ys) > 0:
                    ys = np.vstack(ys)
                    ps = np.vstack(ps)

                    y_mean = np.nanmean(ys, axis=0)
                    y_std = np.nanstd(ys, axis=0)
                    frac_sig = (ps < alpha).sum(0) / ys.shape[0]

                    axs[0, j].plot(t, y_mean, color=c, label=u)
                    axs[0, j].fill_between(
                        t,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=c,
                        alpha=0.25,
                        edgecolor="none",
                    )
                    axs[1, j].plot(t, frac_sig, color=c, alpha=0.7)
                    axs[1, j].fill_between(t, 0, frac_sig, color=c, alpha=0.15)


        # Pseudo-sessions
        for j, k in enumerate(ks):
            for u, c in zip(["visual", "premotor"], ["green", "purple"]):
                ys = []
                ps = []

                for ex in self.experiments["pseudo"][u]:
                    feature = np.asarray(ex.result.feature)
                    time = np.asarray(ex.result.time)
                    statistic = np.asarray(ex.result.statistic)

                    if ex.result.pvalue is None:
                        pvalue_mat = ex.result.compute_pvalues()
                    else:
                        pvalue_mat = ex.result.pvalue

                    unique_features = np.unique(feature)
                    unique_times = np.unique(time)

                    if k not in unique_features:
                        continue

                    i_feature = np.where(unique_features == k)[0][0]

                    y = np.full(len(t), np.nan, dtype=float)
                    p = np.full(len(t), np.nan, dtype=float)

                    mask = feature == k
                    time_k = time[mask]
                    stat_k = statistic[mask]

                    idx = np.argsort(time_k)
                    time_k = time_k[idx]
                    stat_k = stat_k[idx]

                    # Fill statistic curve
                    if np.issubdtype(np.asarray(time_k).dtype, np.integer):
                        valid = (time_k >= 0) & (time_k < len(t))
                        y[time_k[valid]] = stat_k[valid]

                        for tt in time_k[valid]:
                            t_idx = np.where(unique_times == tt)[0][0]
                            p[tt] = pvalue_mat[i_feature, t_idx]
                    else:
                        time_to_index = {tt: i for i, tt in enumerate(t)}
                        for tt, ss in zip(time_k, stat_k):
                            if tt in time_to_index:
                                i_t = time_to_index[tt]
                                y[i_t] = ss
                                t_idx = np.where(unique_times == tt)[0][0]
                                p[i_t] = pvalue_mat[i_feature, t_idx]

                    ys.append(y)
                    ps.append(p)

                if len(ys) > 0:
                    ys = np.vstack(ys)
                    ps = np.vstack(ps)

                    y_mean = np.nanmean(ys, axis=0)
                    y_std = np.nanstd(ys, axis=0)
                    frac_sig = (ps < alpha).sum(0) / ys.shape[0]

                    axs[2, j].plot(t, y_mean, color=c, label=u)
                    axs[2, j].fill_between(
                        t,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=c,
                        alpha=0.25,
                        edgecolor="none",
                    )
                    axs[3, j].plot(t, frac_sig, color=c, alpha=0.7)
                    axs[3, j].fill_between(t, 0, frac_sig, color=c, alpha=0.15)

        # Match y-limits across statistic panels
        ylim = [np.inf, -np.inf]
        for ax in axs[[0, 2], :-1].flatten():
            y1, y2 = ax.get_ylim()
            ylim[0] = min(ylim[0], y1)
            ylim[1] = max(ylim[1], y2)

        y1, y2 = ylim
        for ax in axs[[0, 2], :].flatten():
            ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])

        # Match y-limits across fraction-significant panels
        ylim = [np.inf, -np.inf]
        for ax in axs[[1, 3], :].flatten():
            y1, y2 = ax.get_ylim()
            ylim[0] = min(ylim[0], y1)
            ylim[1] = max(ylim[1], y2)

        y1, y2 = ylim
        for ax in axs[[1, 3], :].flatten():
            ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])

        # Titles
        for ax, title in zip(axs[0, :], titles):
            ax.set_title(title, fontsize=10)

        # Clean y tick labels on non-left panels
        for row in range(4):
            for ax in axs[row, 1:]:
                ax.set_yticklabels([])

        # Row labels
        axs[0, 0].set_ylabel(r"Partial $R^2$")
        axs[1, 0].set_ylabel("Frac. sig.")
        axs[2, 0].set_ylabel(r"Partial $R^2$")
        axs[3, 0].set_ylabel("Frac. sig.")

        fig.supxlabel("Time from saccade initiation (s)", fontsize=10)

        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs