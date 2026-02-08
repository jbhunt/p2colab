import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

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

class PermuationExperiment():
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
        alpha=1.0,
        unit_types=None,
        n_components=1,
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
        y_all = self.ds.filter_X(unit_types)
        N, T, C_2 = y_all.shape
        if C_2 < n_components:
            raise Exception("Less than 3 units in population subset")
        y_all = self.ds.standardize_X(X=y_all)
        y_all = self.ds.decompose_X(X=y_all, n_components=n_components)

        # Build the set of permuted trial indices
        perms = np.empty([n_perms, N], dtype=int)
        for p in range(n_perms):
            perms[p] = np.random.permutation(N)

        # Init model
        # model = Ridge(alpha=alpha, fit_intercept=True)
        model = LinearRegression(fit_intercept=True)

        # Init result object
        self.result = Result()

        # For each kinematic variable
        for j in range(4):

            # For each time bin
            for t in range(T):

                # Identify target
                y = y_all[:, t, :] # (N, C)

                # Partition
                X_0 = np.delete(X_norm, j, axis=1) # Reduced
                X_1 = X_norm # Full

                # Fit reduced model and compute residuals/RSS
                model.fit(X_0, y)
                y_0 = model.predict(X_0)
                E_0 = y - y_0 # NOTE: These are the residuals we will permute later
                rss_0 = np.sum(np.power(E_0, 2))

                # Fit full model and compute residuals/RSS
                model.fit(X_1, y)
                E_1 = y - model.predict(X_1)
                rss_1 = np.sum(np.power(E_1, 2))

                # Compute test statistic (partial R^2)
                partial_r2 = (rss_0 - rss_1) / rss_0

                # Init empirical null dist
                null = np.full(n_perms, np.nan)

                # For each permutation ...
                for p in range(n_perms):

                    # Shuffle the residuals
                    index = perms[p]
                    E_0_shuffled = np.copy(E_0)[index, :]

                    # Create permuted outcome
                    y_null = y_0 + E_0_shuffled

                    # Refit full model
                    model.fit(X_1, y_null)
                    E_1_null = y_null - model.predict(X_1)
                    rss_1_null = np.sum(np.power(E_1_null, 2))

                    # Refit reduced model
                    model.fit(X_0, y_null)
                    E_0_null = y_null - model.predict(X_0)
                    rss_0_null = np.sum(np.power(E_0_null, 2))

                    # Compute test statistic (partial R^2)
                    null[p] = (rss_0_null - rss_1_null) / rss_0_null

                #
                result = Result(j=j, t=t, statistic=partial_r2, null=null)
                self.result.merge_with(result)

        return
    
    def visualize(self, alpha=0.05, figsize=(5, 7)):
        """
        """

        fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True)
        stats = self.result.reshape_statistics()
        nulls = self.result.reshape_nulls()
        t = self.ds.t_X
        M_global = nulls.max(axis=(0, 1)) # Dimensions are K kinematic features (4) x T time bins x P permutations
        threshold = np.quantile(M_global, 1 - alpha)
        for j in range(4):
            axs[j].plot(t, stats[j], color="C0", alpha=0.3)
            masked = np.copy(stats[j])
            masked[masked < threshold] = np.nan
            axs[j].plot(t, masked, color="C0")
            axs[j].hlines(threshold, t.min(), t.max(), color="C0", linestyle=":")
        
        #
        y1, y2 = axs[0].get_ylim()
        for ax in axs:
            ax.vlines(0, y1, y2, color="C0", linestyle=":")
            ax.set_ylim([y1, y2])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        axs[-1].set_xlabel("Time from saccade initiation (s)")
        for i, title in enumerate(["Direction", "Amplitude", "Startpoint", "Endpoint"]):
            axs[i].set_ylabel(title)
        fig.supylabel(r"Partial $R^2$", fontsize=10)
        fig.tight_layout()

        return fig, axs
    
def summarize_v1(experiments, alpha=0.05, figsize=(12, 8)):
    """
    Visualize partial R^2 across time for each kinematic feature for each session/experiment
    """

    fig, axs = plt.subplots(nrows=4, ncols=len(experiments), sharex=True, sharey=True)

    #
    t = experiments[0].ds.t_X # NOTE: Assuming that all experiments used the same time window
    all_curves = np.full([4, len(experiments), t.size], np.nan)

    #
    for j, ex in enumerate(experiments):
        
        #
        stats = ex.result.reshape_statistics()
        nulls = ex.result.reshape_nulls()
        M_global = nulls.max(axis=(0, 1))
        threshold = np.quantile(M_global, 1 - alpha)

        #
        for i in range(4):
            axs[i, j].plot(t, stats[i], color="C0", alpha=0.3)
            masked = np.copy(stats[i])
            masked[masked < threshold] = np.nan
            all_curves[i, j] = masked
            axs[i, j].plot(t, masked, color="C0")
            axs[i, j].hlines(threshold, t.min(), t.max(), color="C0", linestyle=":")
        
    # Plot vertical lines
    y1, y2 = axs[0, 0].get_ylim()
    for ax in axs.flatten():
        ax.vlines(0, y1, y2, color="C0", linestyle=":")
        ax.set_ylim([y1, y2])

    #
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.supxlabel("Time from saccade initiation (s)", fontsize=10)
    for i, title in enumerate(["Direction", "Amplitude", "Startpoint", "Endpoint"]):
        axs[i, 0].set_ylabel(title)
    fig.supylabel(r"Partial $R^2$", fontsize=10)
    fig.tight_layout()

    return fig, axs

def summarize_v2(experiments, alpha=0.05, figsize=(4, 6)):
    """
    Aggregate visualization over all sessions/experiments
    """

    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True)

    #
    t = experiments[0].ds.t_X # NOTE: Assuming that all experiments used the same time window
    all_curves = np.full([4, len(experiments), t.size], np.nan)

    #
    for j, ex in enumerate(experiments):
        
        #
        stats = ex.result.reshape_statistics()
        nulls = ex.result.reshape_nulls()
        M_global = nulls.max(axis=(0, 1))
        threshold = np.quantile(M_global, 1 - alpha)

        #
        for i in range(4):
            # axs[i].plot(t, stats[i], color="C0", alpha=0.3)
            masked = np.copy(stats[i])
            masked[masked < threshold] = np.nan
            all_curves[i, j] = masked

    # Overlay curves for all sessions
    for i in range(4):
        ax = axs[i]
        for j in range(len(experiments)):
            y = all_curves[i, j]
            ax.plot(t, y, color="C0", alpha=0.2)
    
    #
    for i in range(4):
        axs[i].plot(t, np.nanmean(all_curves[i], axis=0), color="C0")

    # Plot vertical lines
    y1, y2 = axs[0].get_ylim()
    for ax in axs:
        ax.vlines(0, y1, y2, color="C0", linestyle=":")
        ax.set_ylim([y1, y2])

    #
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.supxlabel("Time from saccade initiation (s)", fontsize=10)
    for i, title in enumerate(["Direction", "Amplitude", "Startpoint", "Endpoint"]):
        axs[i].set_ylabel(title)
    fig.supylabel(r"Partial $R^2$", fontsize=10)
    fig.tight_layout()

    return fig, axs