import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from .datasets import SyntheticMlatiDataset, LazyMergedSessions

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

def compute_partial_statistics():
    """
    Computes the partial R^2 and F statistics

    TODO: Encapsulate these computations here
    """

    return

class Result():
    """
    """

    def __init__(self, j=None, t=None, statistic=None, null=None):
        """
        """

        self._j = j
        self._t = t
        self._statistic = statistic
        self._null = null

        return
    
    def merge_with(self, r):
        """
        """

        attrs = ("_j", "_t", "_statistic")
        for attr in attrs:
            a = getattr(self, attr)
            b = getattr(r, attr)
            if a is None:
                a = np.empty([0])
            if b is None:
                b = np.empty([0])
            if np.isscalar(a):
                a = np.atleast_1d(a)
            if np.isscalar(b):
                b = np.atleast_1d(b)
            merged = np.concatenate([a, b])
            self.__setattr__(attr, merged)

        #
        a = self.null
        b = r.null
        if a is None:
            merged = b
        else:
            merged = np.vstack([a, b])
        self._null = merged

        return
    
    def reshape_statistics(self):
        """
        """

        n_j = len(np.unique(self.j))
        n_t = len(np.unique(self.t))
        statisitcs = self.statistic.reshape(n_j, n_t)

        return statisitcs
    
    def reshape_nulls(self):
        """
        """

        n_j = len(np.unique(self.j))
        n_t = len(np.unique(self.t))
        nulls = self.null.reshape(n_j, n_t, -1)

        return nulls
    
    def compute_pvalues(self):
        """
        Adjust p-values for multiple comparisons across features and time
        """

        stats = self.reshape_statistics()
        nulls = self.reshape_nulls()
        n_j, n_t, n_p = nulls.shape
        M_perm = nulls.max(axis=(0, 1))
        ge = (M_perm[None, None, :] >= stats[:, :, None])
        p_adjusted = (1 + ge.sum(axis=2)) / (1 + n_p)

        return p_adjusted
    
    def compute_pvalues_v2(self):
        """
        Adjust p-values for multiple comparisons across time (but not features)
        """

        # TODO: Implement this method

        return
    
    @property
    def j(self):
        return self._j
    
    @property
    def t(self):
        return self._t
    
    @property
    def statistic(self):
        return self._statistic
    
    @property
    def null(self):
        return self._null

class SingleSessionPermuationExperiment():
    """
    """

    def __init__(self, ds):
        """
        """

        self.ds = ds
        self.result = None

        return
    
    def run(
        self,
        n_perms=100,
        alpha=0.0,
        n_components=10,
        unit_types=["premotor", "visuomotor"],
        ):
        """
        """

        #
        N, T, C_1 = self.ds.X.shape

        # Build baseline design matrix
        X = np.vstack([
            self.ds.saccade_direction,
            self.ds.saccade_amplitude,
            self.ds.saccade_startpoints,
            self.ds.saccade_endpoints
        ]).T
        X_norm = StandardScaler().fit_transform(X)

        # Transform target
        y = self.ds.filter_X(unit_types)
        N, T, C_2 = y.shape # (N trials, T time bins, C components/units)
        y = self.ds.standardize_X(X=y)
        if n_components is not None:
            if C_2 < n_components:
                raise Exception("# of units is less than the number of PCs")
            y = self.ds.decompose_X(X=y, n_components=n_components)

        # Build the set of permuted trial indices (permutations only happen within blocks)
        unique_blocks = np.unique(self.ds.saccade_blocks)
        block_indices = {b: np.where(self.ds.saccade_blocks == b)[0] for b in unique_blocks}
        perms = np.empty([n_perms, N], dtype=int)
        for p in range(n_perms):
            perm = np.empty(N, dtype=int)
            for b, idx in block_indices.items():
                perm[idx] = np.random.permutation(idx)
            perms[p] = perm

        # Init model
        if alpha == 0:
            model = LinearRegression(fit_intercept=True) # Use OLS
        else:
            model = Ridge(alpha=alpha, fit_intercept=True)

        # Init result object
        self.result = Result()

        #
        n_runs = 4 * T * n_perms
        i_run = 0

        # For each kinematic variable
        for j in range(4):

            # For each time bin
            for t in range(T):

                # Identify target time bin
                y_t = y[:, t, :] # (N, C)

                # Partition
                X_0 = np.delete(X_norm, j, axis=1) # Reduced
                X_1 = X_norm # Full

                # Fit reduced model and compute residuals/RSS
                model.fit(X_0, y_t)
                y_0 = model.predict(X_0)
                E_0 = y_t - y_0 # NOTE: These are the residuals we will permute later
                rss_0 = np.sum(np.power(E_0, 2))

                # Fit full model and compute residuals
                model.fit(X_1, y_t)
                E_1 = y_t - model.predict(X_1)
                rss_1 = np.sum(np.power(E_1, 2))

                # Compute test statistic (partial R^2)
                partial_r2 = (rss_0 - rss_1) / rss_0

                # Init empirical null dist
                null = np.full(n_perms, np.nan)

                # For each permutation ...
                for p in range(n_perms):

                    #
                    end = "\r" if (i_run + 1) < n_runs else "\n"
                    print(f"Running fit {i_run + 1} out of {n_runs}", end=end)

                    # Shuffle the residuals
                    index = perms[p]
                    E_0_shuffled = np.copy(E_0)[index, :]

                    # Create permuted outcome
                    y_null = y_0 + E_0_shuffled

                    # Refit full model
                    model.fit(X_1, y_null)
                    E_1_null = y_null - model.predict(X_1)
                    rss_1_null = np.sum(np.power(E_1_null, 2))

                    # Compute test statistic (partial R^2)
                    null[p] = (rss_0 - rss_1_null) / rss_0

                    #
                    i_run += 1

                #
                result = Result(j=j, t=t, statistic=partial_r2, null=null)
                self.result.merge_with(result)

        return
    
    def visualize(self, alpha=0.05, color="C0",figsize=(5, 7), axs=None):
        """
        """

        if axs is None:
            fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True)
        else:
            fig = axs[0].figure
        stats = self.result.reshape_statistics()
        nulls = self.result.reshape_nulls()
        pvals = self.result.compute_pvalues()
        t = self.ds.t_X
        M_global = nulls.max(axis=(0, 1)) # Dimensions are K kinematic features (4) x T time bins x P permutations
        threshold = np.quantile(M_global, 1 - alpha)
        for j in range(4):
            y = stats[j]
            axs[j].plot(t, y, color=color, alpha=0.3)
            f = interp1d(t, y)
            t_eval = np.linspace(t.min(), t.max(), 1000 + 1)
            y_eval = f(t_eval)
            y_eval[y_eval < threshold] = np.nan
            axs[j].plot(t_eval, y_eval, color=color, alpha=0.7)
            # axs[j].hlines(threshold, t.min(), t.max(), color="k", lw=1.0, linestyle=":")
        
        #
        ylim = [np.inf, -np.inf]
        for ax in axs:
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        y1, y2 = ylim
        for ax in axs:
            # ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])

        #
        if axs is None:
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])
            axs[-1].set_xlabel("Time from saccade initiation (s)")
            for i, title in enumerate(["Direction", "Amplitude", "Startpoint", "Endpoint"]):
                axs[i].set_ylabel(title)
            fig.supylabel(r"Partial $R^2$", fontsize=10)
            fig.tight_layout()

        return fig, axs
    
class AllPermutationExperiments():
    """
    """

    def __init__(self, sessions):
        """
        """

        self.sessions = sessions
        self.experiments = None

        return
    
    def run(self, n_components=3, *args, **kwargs):
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
            # ["visual", "visuomotor", "premotor"],
            None
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
                ex = SingleSessionPermuationExperiment(ds)
                ex.run(unit_types=s, n_components=n_components, *args, **kwargs)
                self.experiments["s"][k].append(ex)

            #
            ds = LazyMergedSessions(sessions)
            ex = SingleSessionPermuationExperiment(ds)
            ex.run(unit_types=s, n_components=n_components, *args, **kwargs)
            self.experiments["m"][k].append(ex)

        return

    def _visualize_single_session_results(self, alpha=0.05, figsize=(8, 5)):
        """
        """

        fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True)

        #
        t = self.experiments["s"]["vo"][0].ds.t_X # NOTE: Assuming that all experiments used the same time window

        #
        for (j, s, c) in zip([0, 1, 2, 3], ["vo", "vm", "pm", "nf"], ["C0", "C1", "C2", "C3"]):
            
            #
            all_curves = np.full([4, len(self.sessions), t.size], np.nan)
            for k, ex in enumerate(self.experiments["s"][s]):
                
                #
                stats = ex.result.reshape_statistics()
                nulls = ex.result.reshape_nulls()
                pvals = ex.result.compute_pvalues()
                M_global = nulls.max(axis=(0, 1))
                threshold = np.quantile(M_global, 1 - alpha)

                #
                for i in range(4):
                    masked = np.copy(stats[i])
                    # masked[pvals[i] >= alpha] = np.nan
                    all_curves[i, k] = masked

            # Overlay curves for all sessions
            for i in range(4):

                #
                ax = axs[i, j]
                # y = (~np.isnan(all_curves[i])).sum(0) / len(self.sessions)
                ys = all_curves[i]
                for y in ys:
                    ax.plot(t, y, color=c, lw=1.0, alpha=0.2)
                ax.plot(t, np.nanmean(ys, axis=0), color=c)

        # Left block
        ylim = [np.inf, -np.inf]
        for ax in axs.flatten():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        y1, y2 = ylim
        for ax in axs[:, :4].flatten():
            # ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])

        #
        fig.supylabel(r"Partial $R^2$", fontsize=10)
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
    
    def _visualize_multi_session_results(self, alpha=0.05, figsize=(8, 5)):
        """
        """

        fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True)

        #
        t = self.experiments["s"]["vo"][0].ds.t_X # NOTE: Assuming that all experiments used the same time window

        #
        for (j, s, c) in zip([0, 1, 2, 3], ["vo", "vm", "pm", "nf"], ["C0", "C1", "C2", "C3"]):
            ex = self.experiments["m"][s][0]
            _ = ex.visualize(axs=axs[:, j], color=c)

        # Left block
        ylim = [np.inf, -np.inf]
        for ax in axs.flatten():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        y1, y2 = ylim
        for ax in axs[:, :4].flatten():
            ax.vlines(0, y1, y2, color="k", linestyle=":", lw=1.0)
            ax.set_ylim([y1, y2])

        #
        fig.supylabel(r"Partial $R^2$", fontsize=10)
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
    
    def visualize(self, alpha=0.05, figsize=(8, 5)):
        """
        """

        fig1, _ = self._visualize_single_session_results(alpha=alpha, figsize=figsize)
        fig2, _ = self._visualize_multi_session_results(alpha=alpha, figsize=figsize)
        figs = [fig1, fig2]

        return figs

class ControlExperiment():
    """
    """

    def __init__(self, ds=None, coeffs=np.linspace(0, 8, 11)):
        """
        """

        self.ds = None
        self.coeffs = coeffs
        self.result = None
        self.n_runs = None
        self.eps_noise_X = None
        self.eps_noise_y = None

        return
    
    def run(self, n_runs=10, n_trials=100, eps_noise_y=0.25, eps_noise_X=0):
        """
        """

        self.n_runs = n_runs
        self.eps_noise_y = eps_noise_y
        self.eps_noise_X = eps_noise_X
        self.result = {
            "stats": np.full([len(self.coeffs), n_runs, 4], np.nan),
            "pvalues": np.full([len(self.coeffs), n_runs, 4], np.nan)
        }
        for i, coeff in enumerate(self.coeffs):
            for j in range(n_runs):
                ds = SyntheticMlatiDataset(n_trials=n_trials, regime=2, rho_within=0.7, rho_between=coeff, eps_noise_X=eps_noise_X, eps_noise_y=eps_noise_y)
                ex = SingleSessionPermuationExperiment(ds)
                ex.run()
                stats = ex.result.reshape_statistics()
                pvalues = ex.result.compute_pvalues()
                for k in range(4):
                    self.result["stats"][i, j, k] = stats[k].item()
                    self.result["pvalues"][i, j, k] = pvalues[k].item()

        return
    
    def visualize(self, xs=[0.1, 0.5]):
        """
        """

        fig, axs = plt.subplots(ncols=2, nrows=2, gridspec_kw={"height_ratios": [1, 3]})
        ds_1 = SyntheticMlatiDataset(n_trials=1000, regime=2, rho_within=0.7, rho_between=xs[0], eps_noise_y=self.eps_noise_y)
        ds_2 = SyntheticMlatiDataset(n_trials=1000, regime=2, rho_within=0.7, rho_between=xs[1], eps_noise_y=self.eps_noise_y)
        axs[0, 0].imshow(np.corrcoef(ds_1.inputs, rowvar=False), vmin=0, vmax=1, aspect="auto")
        axs[0, 1].imshow(np.corrcoef(ds_2.inputs, rowvar=False), vmin=0, vmax=1, aspect="auto")
        for k in range(4):
            axs[1, 0].plot(self.coeffs, self.result["stats"][:, :, k].mean(1))
            n_sig = np.sum(self.result["pvalues"][:, :, k] < 0.05, axis=1)
            n_sig = n_sig / self.n_runs
            axs[1, 1].plot(self.coeffs, n_sig)
        y1, y2 = axs[1, 0].get_ylim()
        axs[1, 0].vlines(xs, y1, y2, color="k", linestyle=":")
        axs[1, 0].set_ylim([y1, y2])
        y1, y2 = axs[1, 1].get_ylim()
        axs[1, 1].vlines(xs, y1, y2, color="k", linestyle=":")
        axs[1, 1].set_ylim([y1, y2])
        axs[0, 0].set_title(r"$\rho_{within}=0.7, \rho_{between}$=" + f"{xs[0]}", fontsize=10)
        axs[0, 1].set_title(r"$\rho_{within}=0.7, \rho_{between}$=" + f"{xs[1]}", fontsize=10)
        axs[0, 0].set_xticks([0, 1, 2, 3])
        axs[0, 0].set_xlabel("Regressors")
        axs[0, 0].set_yticks([0, 1, 2, 3])
        axs[0, 0].set_ylabel("Regressors")
        axs[1, 0].set_xlabel(r"$\rho_{between}$")
        axs[1, 0].set_ylabel(r"Partial $R^2$")
        axs[1, 1].set_ylabel("Frac. significant runs")
        fig.tight_layout()

        return fig, axs 