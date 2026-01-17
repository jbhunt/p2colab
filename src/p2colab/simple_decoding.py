import numpy as np
from .utils import MlatiSessionDataset
from .models import Seq2SeqDecoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

class SaccadeWaveformScaler():
    """
    """

    def __init__(self):
        """
        """

        self.scaler = StandardScaler()

        return
    
    def fit(self, y):
        """
        """

        self.scaler.fit(y)

        return self
    
    def transform(self, y):
        """
        """

        return self.scaler.transform(y)
    
    def untransform(self, y):
        """
        """

        return self.scaler.inverse_transform(y)


class Result():
    """
    """

    _statistic = None
    _null = None
    _d = None
    _p = None

    def __init__(self, statistic, null, y_true=None, y_pred=None, z_true=None):
        """
        """

        self._statistic = round(statistic, 3)
        self._null = null
        self._p = round(np.sum(null <= statistic) / len(null), 3).item()
        self._d = round((statistic - null.mean()) / null.std(), 3).item()
        self._y_pred = y_pred
        self._y_true = y_true
        self._z_true = z_true
        self._residuals = y_true - y_pred
        
        # Compute coeff. of det. per waveform
        tss = np.sum((y_true - y_true.mean(1, keepdims=True)) ** 2, axis=1)
        rss = np.sum((y_true - y_pred) ** 2, axis=1)
        self._r2 = 1 - rss / tss

        return

    @property
    def statistic(self):
        return self._statistic
    
    @property
    def null(self):
        return self._null

    @property
    def d(self):
        return self._d
    
    @property
    def p(self):
        return self._p
    
    @property
    def y_pred(self):
        return self._y_pred
    
    @property
    def y_true(self):
        return self._y_true
    
    @property
    def z_true(self):
        return self._z_true
    
    @property
    def residuals(self):
        return self._residuals
    
    @property
    def r2(self):
        return self._r2

class SimpleDecodingExperiment():
    """
    Test how well I'm able to decode eye position from premotor (pre-saccade initiation) neural acitivity
    """

    def __init__(self, ds):
        """
        """

        self.ds = ds
        self.est = None
        self.loss = None
        self.result = None

        return
    
    def run(
        self,
        n_runs=30,
        train_size=0.8,
        validation_size=0.1,
        kernel_size=5,
        lr=0.0001,
        max_iter=500,
        batch_size=32,
        split_seed=42,
        ):
        """
        """

        # Make splits
        ds_train, ds_test = self.ds.random_split([train_size, 1 - train_size], split_seed=split_seed)
        ds_train, ds_valid = ds_train.random_split([1 - validation_size, validation_size], split_seed=split_seed)

        # Init model
        _, _, U = ds_train.X.shape
        P = ds_train.saccade_waveforms.shape[-1]
        self.est = Seq2SeqDecoder(
            U,
            P,
            kernel_size=kernel_size,
            lr=lr,
            max_iter=max_iter,
            batch_size=batch_size
        )

        # Scale eye position data
        f_y = SaccadeWaveformScaler().fit(ds_train.y)
        for ds in (ds_train, ds_valid, ds_test):
            y_tr = f_y.transform(ds.y)
            ds.set_y(y_tr)

        #
        print(f"Working on run 1 out of {n_runs + 1}")
        self.est.fit(ds_train, ds_valid, print_info=True)
        y_pred = self.est.predict(ds_test)
        X_test, y_test = ds_test[:]
        rmse = round(np.sqrt(np.mean(np.power(y_test.flatten() - y_pred.flatten(), 2))).item(), 6)

        # Run decoding permutation test to estimate chance level of performance
        null = np.full(n_runs, np.nan)
        for i_run in range(n_runs):

            #
            print(f"Working on run {i_run + 2} out of {n_runs + 1}")

            # Shuffle trials
            new_y = ds_train.y.copy()
            shuffled_indices = np.random.choice(np.arange(len(new_y)), size=len(new_y), replace=False)
            new_y = new_y[shuffled_indices]
            ds_train.set_y(new_y)

            # Refit
            self.est._return_to_initial_state()
            self.est.fit(ds_train, ds_valid, print_info=True)

            # Eval
            y_pred_ = self.est.predict(ds_test)
            rmse_ = round(np.sqrt(np.mean(np.power(y_test.flatten() - y_pred_.flatten(), 2))).item(), 6)
            null[i_run] = rmse_

        #
        y_test = f_y.untransform(y_test)
        y_pred = f_y.untransform(y_pred)
        self.result = Result(rmse, null, y_test, y_pred, z_true=ds_test.saccade_direction)

        return
    
    def visualize(self, figsize=(7, 5), v_abs=30, t=None):
        """
        """

        height_ratios = [
            np.sum(self.result.z_true == 0),
            np.sum(self.result.z_true == 1)
        ]
        fig, axs = plt.subplots(
            nrows=2, ncols=4,
            gridspec_kw={
                "width_ratios": [4, 4, 4, 1],
                "height_ratios": height_ratios
            }
        )

        #
        y_residuals = self.result.y_true - self.result.y_pred
        X = self.ds.t_y
        for i, z in enumerate([0, 1]):
            Y = np.arange(height_ratios[i])
            index_1 = np.where(self.result.z_true == z)[0]
            r2 = np.clip(self.result.r2.reshape(-1, 1)[index_1], 0, 1)
            index_2 = np.argsort(r2.flatten())
            index = index_1[index_2]
            for j, Z in enumerate([self.result.y_true, self.result.y_pred, y_residuals]):
                Z = Z[index, :]
                Z_sub = Z - Z[:, :10].mean(axis=1, keepdims=True)
                axs[i, j].pcolor(X, Y, Z, cmap="binary_r",vmin=-v_abs, vmax=v_abs)
            axs[i, 3].pcolor(r2[index_2, :], cmap="binary_r", vmin=0, vmax=1)

        #
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[0, :].flatten():
            ax.set_xticklabels([])
        for ax in axs[:, :-1].flatten():
            y1, y2 = ax.get_ylim()
            ax.vlines(0, y1, y2, color="k", linestyle=":")
            ax.set_ylim([y1, y2])
        axs[-1, 0].set_xlabel("Time from saccade initiation (s)")
        axs[-1, 0].set_ylabel("Saccade index")
        axs[-1, -1].yaxis.set_label_position("right")
        axs[-1, -1].yaxis.tick_right()
        axs[-1, -1].set_ylabel(r"$R^{2}$", rotation=270, labelpad=20)
        axs[0, 0].set_title(r"$Y_{True}$", fontsize=10)
        axs[0, 1].set_title(r"$Y_{Pred.}$", fontsize=10)
        axs[0, 2].set_title(r"$Y_{Res.}$", fontsize=10)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs
