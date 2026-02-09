import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import polars as pl
from pathlib import Path
from datetime import date

class MlatiSessionDataset(Dataset):
    """
    """

    def __init__(self, src, lut=None, **kwargs):
        """
        """

        super().__init__()
        self.src = Path(src)
        date_, animal, _ = self.src.stem.split("_")
        self.date = date.fromisoformat(date_)
        self.animal = animal
        self.lut = lut
        self._unit_ids = None
        self._X = None
        self._X_raw = None
        self._y = None
        self._t = None
        self._saccade_waveforms = None
        self._saccade_direction = None
        self._saccade_startpoints = None
        self._saccade_endpoints = None
        self._saccade_amplitude = None
        self._overrides = {}
        self._index = None
        self._loaded = False

        #
        kwargs_ = {
            "derivative": 0,
            "X_binsize": 0.02,
            "X_bincounts": (25, 0),
            "y_binsize": 0.002,
            "y_bincounts": (25, 45),
            "p_max": None,
        }
        kwargs_.update(kwargs)
        self._load(**kwargs_)

        return
    
    def _apply_index(self, arr):
        """
        """

        if self._index is None:
            return arr
        else:
            return arr[self._index]
    
    def set_override(self, name, value):
        """
        """

        value = np.asarray(value)
        if len(value) != len(self):
            raise Exception("New value must have the same length as dataset")
        self._overrides[name] = value

        return
    
    def clear_overrides(self):
        """
        """

        self._overrides.clear()

        return
    
    def _load(
        self,
        derivative=0,
        X_binsize=0.02,
        X_bincounts=(25, 0),
        y_binsize=0.002,
        y_bincounts=(25, 45),
        p_max=None,
        ):
        """
        """

        #
        if self.loaded:
            raise Exception("Dataset is already loaded")

        # Load all the required datasets from the h5 file
        with h5py.File(str(self.src), 'r') as stream:
            eye_position = np.array(stream['pose/filtered'])[:, 0]
            n_frames_recorded = len(eye_position)
            frame_timestamps = np.array(stream['frames/left/timestamps'])[:n_frames_recorded]
            spike_timestamps = np.array(stream[f'spikes/timestamps'])
            spike_clusters = np.array(stream[f'spikes/clusters'])
            p_values = np.vstack([
                np.array(stream['zeta/saccade/nasal/p']),
                np.array(stream['zeta/saccade/temporal/p'])
            ]).min(0)
            saccade_timestamps = np.array(stream['saccades/predicted/left/timestamps'])
            saccade_labels = np.array(stream['saccades/predicted/left/labels'])

        # For some experiments there are different numbers of frames and timestamps which will preclude further processing
        if frame_timestamps.size != eye_position.size:
            raise Exception(f'Different number of frames ({eye_position.size}) and frame timestamps ({frame_timestamps.size})')
        
        # Drop saccades without timestamps
        invalid_saccades = np.isnan(saccade_timestamps).any(1)
        saccade_labels = np.delete(saccade_labels, invalid_saccades)
        saccade_timestamps = np.delete(saccade_timestamps, invalid_saccades, axis=0)
        saccade_onset_timestamps = saccade_timestamps[:, 0]
        saccade_offset_timestamps = saccade_timestamps[:,1]

        # Option to add other events than saccades
        event_timestamps = np.concatenate([saccade_onset_timestamps,])
        z = np.concatenate([saccade_labels,])
        z[saccade_labels == -1] = 0

        # Sort by time
        index = np.argsort(event_timestamps)
        event_timestamps = event_timestamps[index]
        z = z[index]

        # Create the eye position time series
        t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
        if derivative == 0:
            t_raw = frame_timestamps
            y_raw = eye_position
        elif derivative == 1:
            t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
            y_raw = np.diff(eye_position) / y_binsize
        else:
            raise Exception(f'Derivatives > 1 not supported')
        y_raw[np.isnan(y_raw)] = np.interp(t_raw[np.isnan(y_raw)], t_raw, y_raw) # Impute with interpolation

        # Collect the eye velocity waveforms for saccades 
        # TODO: Align eye position with neural activity (use the same bins)
        saccade_waveforms = list()
        t_eval = y_binsize * (np.arange(-1 * y_bincounts[0], y_bincounts[1], 1) + 0.5)
        self._t_y = t_eval
        for event_timestamp in event_timestamps:
            wf = np.interp(
                t_eval + event_timestamp,
                t_raw,
                y_raw
            )

            # Need to scale signal if derivative = 1
            if derivative == 1: # TODO: Check if this is right
                wf =  wf / y_binsize
            saccade_waveforms.append(wf)
        saccade_waveforms = np.array(saccade_waveforms)

        #
        saccade_startpoints = list()
        saccade_endpoints = list()
        for t1, t2 in zip(saccade_onset_timestamps, saccade_offset_timestamps):
            p1 = np.interp(t1, t_raw, y_raw)
            p2 = np.interp(t2, t_raw, y_raw)
            saccade_startpoints.append(p1)
            saccade_endpoints.append(p2)
        saccade_startpoints = np.array(saccade_startpoints)
        saccade_endpoints = np.array(saccade_endpoints)
        saccade_amplitude = np.abs(saccade_endpoints - saccade_startpoints)

        # Exclude units without event-related activity
        unique_clusters = np.unique(spike_clusters)
        if p_max is None:
            target_clusters = unique_clusters
        else:
            cluster_indices = np.arange(len(unique_clusters))[p_values <= p_max]
            target_clusters = unique_clusters[cluster_indices]

        #
        self._unit_ids = target_clusters

        # Compute the edges of the time bins centered on the saccade
        left_edges = np.arange(-1 * X_bincounts[0], X_bincounts[1], 1)
        right_edges = left_edges + 1
        try:
            all_edges = np.concatenate([left_edges, [right_edges[-1],]]) * X_binsize
        except:
            import pdb; pdb.set_trace()
        self._t_X = all_edges[:-1] + (X_binsize / 2)

        # Compute histograms and store in response matrix of shape N units x M saccades x P time bins
        n_units = len(target_clusters)
        R = list()
        for i_unit, target_cluster in enumerate(target_clusters):
            end = '\r' if i_unit + 1 != n_units else '\n'
            print(f'Computing histograms for unit {i_unit + 1} out of {n_units} ...', end=end)
            spike_indices = np.where(spike_clusters == target_cluster)[0]
            sample = list()
            for event_timestamp in event_timestamps:
                n_spikes, bin_edges_ = np.histogram(
                    spike_timestamps[spike_indices],
                    bins=np.around(all_edges + event_timestamp, 3)
                )
                # fr = n_spikes / X_binsize
                sample.append(n_spikes.astype(np.float32))
            R.append(sample)
        R = np.array(R) # U x N x T
        X = np.swapaxes(R, 0, 1) # N x U x T
        self._X = np.swapaxes(X, 1, 2) # N x T x U
        self._X_raw = np.copy(self._X)
        self._y = saccade_waveforms # Default target is saccade waveforms (but this can be overriden with the set_y method)
        self._y_raw = np.copy(self._y)
        self._saccade_waveforms = saccade_waveforms
        self._saccade_startpoints = saccade_startpoints
        self._saccade_endpoints = saccade_endpoints
        self._saccade_amplitude = saccade_amplitude
        self._saccade_direction = z

        #
        self._loaded = True

        return
    
    def resample_saccade_waveforms(self, T_out):
        """
        """

        X_in = self.saccade_waveforms
        N, T_in = X_in.shape
        X_out = np.full([N, T_out], np.nan)
        t_eval = np.linspace(self.t_y.min(), self.t_y.max(), T_out)
        for i in range(N):
            X_out[i] = np.interp(t_eval, self.t_y, X_in[i])

        return X_out
    
    @property
    def t_X(self):
        return self._t_X
    
    @property
    def t_y(self):
        return self._t_y
    
    def compress_X(self, fn=np.log1p):
        """
        """

        X_out = fn(self._X)

        return
    
    def standardize_X(self, X=None):
        """
        """

        # X has shape N x T x U
        X = self._X if X is None else X
        mean_fr = X.mean(axis=(0, 1), keepdims=True) # 1 x 1 x U
        std_fr = X.std(axis=(0, 1), keepdims=True) + 1e-8 # Same shape as mean
        X_out = (X - mean_fr) / std_fr

        return X_out
    
    def filter_X(self, unit_types=("premotor", "visuomotor", "visual")):
        """
        Return a view of X for a subset of unit types
        """

        if unit_types is None:
            return self.X

        if self.lut is None:
            raise Exception("Lookup table not specificed at instantiation")
        
        lut = pl.read_csv(self.lut).with_columns(
            pl.col("date").str.to_date("%m/%d/%Y")
        )
        lut = lut.filter((pl.col("date") == self.date) & (pl.col("animal") == self.animal) & (pl.col("utype").is_in(unit_types)))
        target_unit_ids = lut.select("uid").to_numpy().flatten()
        _, index, _ = np.intersect1d(self.unit_ids, target_unit_ids, return_indices=True)
        X_out = self.X[:, :, index]

        return X_out
    
    def decompose_X(self, n_components=3, X=None):
        """
        """

        pca = PCA(n_components=n_components)
        if X is None:
            X = self.X
        N, T, C = X.shape
        X_in = X.reshape(N * T, C)
        pca.fit(X_in)
        X_out = pca.transform(X_in)
        X_out = X_out.reshape(N, T, n_components)

        return X_out
    
    def set_X(self, X):
        """
        """

        self._X = X

        return
    
    def reset_X(self):
        """
        """

        if "X" in self._overrides.keys():
            del self._overrides["X"]
        self._X = np.copy(self._X_raw)

        return
    
    def set_y(self, y):
        """
        """

        self._y = y

        return
    
    def reset_y(self):
        """
        """

        if "y" in self._overrides.keys():
            del self._overrides["y"]
        self._y = np.copy(self._y_raw)

        return
    
    def reset_Xy(self):
        """
        """

        if "X" in self._overrides.keys():
            del self._overrides["X"]
        self._X = np.copy(self._X_raw)
        if "y" in self._overrides.keys():
            del self._overrides["y"]
        self._y = np.copy(self._y_raw)

        return
    
    def spawn_subset(self, indices, copy=True):
        """
        """

        ds = self.__class__.__new__(self.__class__)
        Dataset.__init__(ds)
        ds._overrides = {}
        ds._loaded = True
        ds.src = self.src
        ds.lut = self.lut

        # Establish new index
        if self._index is None:
            new_index = indices
        else:
            new_index = self._index[indices]

        if copy:
            ds._X = self._X[new_index].copy()
            ds._X_raw = self._X_raw[new_index].copy()
            ds._y = self._y[new_index].copy()
            ds._y_raw = self._y_raw[new_index].copy()
            ds._saccade_waveforms = self._saccade_waveforms[new_index].copy()
            ds._saccade_amplitude = self._saccade_amplitude[new_index].copy()
            ds._saccade_direction = self._saccade_direction[new_index].copy()
            ds._saccade_startpoints = self._saccade_startpoints[new_index].copy()
            ds._saccade_endpoints = self._saccade_endpoints[new_index].copy()
            ds._index = None
        else:
            ds._X = self._X
            ds._X_raw = self._X_raw
            ds._y = self._y
            ds._y_raw = self._y_raw
            ds._saccade_waveforms = self._saccade_waveforms
            ds._saccade_amplitude = self._saccade_amplitude
            ds._saccade_direction = self._saccade_direction
            ds._saccade_startpoints = self._saccade_startpoints
            ds._saccade_endpoints = self._saccade_endpoints
            ds._index = new_index

        return ds
    
    def random_split(self, split_sizes=[0.8, 0.2], split_seed=42):
        """
        """

        #
        if bool(np.isclose(sum(split_sizes), 1.0, rtol=1e-3)) == False:
            raise Exception("Split sizes must sum to 1")

        #
        n_total = len(self)
        counts = [int(round(s * n_total)) for s in split_sizes]
        counts[-1] = n_total - sum(counts[:-1])

        #
        g = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(n_total, generator=g)

        # Build split indices
        splits = list()
        i_start = 0
        for c in counts:
            idx = perm[i_start: i_start + c]
            split = self.spawn_subset(idx, copy=True)
            splits.append(split)
            i_start += c

        return splits
    
    def split_on_Z(self):
        """
        Split into 2 subsets based on saccade direction
        """

        subsets = []
        for z in (0, 1):
            idx = np.where(self.saccade_direction == z)[0]
            subset = self.spawn_subset(idx, copy=True)
            subsets.append(subset)

        return subsets
    
    @property
    def loaded(self):
        return self._loaded
    
    @property
    def X(self):
        return self._apply_index(self._overrides.get("X", self._X))
    
    @property
    def y(self):
        return self._apply_index(self._overrides.get("y", self._y))
    
    @property
    def saccade_waveforms(self):
        return self._apply_index(self._overrides.get("saccade_waveforms", self._saccade_waveforms))
    
    @property
    def saccade_direction(self):
        return self._apply_index(self._overrides.get("saccade_direction", self._saccade_direction))
    
    @property
    def saccade_amplitude(self):
        return self._apply_index(self._overrides.get("saccade_amplitude", self._saccade_amplitude))
    
    @property
    def saccade_startpoints(self):
        return self._apply_index(self._overrides.get("saccade_startpoints", self._saccade_startpoints))

    @property
    def saccade_endpoints(self):
        return self._apply_index(self._overrides.get("saccade_endpoints", self._saccade_endpoints))
    
    @property
    def unit_ids(self):
        return self._unit_ids
    
    def __len__(self):
        """
        """

        if self._index is None:
            return self._X.shape[0]
        else:
            return len(self._index)
    
    def __getitem__(self, index):
        X_i = self.X[index]
        y_i = self.y[index]
        return X_i, y_i
    
class SyntheticMlatiDataset(Dataset):
    """
    """

    def __init__(self, n_trials=1, regime=1, rho=0.7, eps=0.25, cor=0.0, n_X=2):
        """
        inputs
        ------
        n_trials
            Number of simulated trials
        regime
            Experiment regime
        rho
            Correlation between nuisance variables
        eps
            Scale of noise added to target
        cor
            Correlation between signal and nuisance variables
        """

        self.n_trials = n_trials
        self.n_X = n_X
        self.rho = rho
        self.eps = eps
        self.cor = cor
        self.regime = regime
        self._X = None
        self._inputs = None
        self._output = None
        self._saccade_direction = None
        self._saccade_amplitude = None
        self._saccade_startpoints = None
        self._saccade_endpoints = None
        if self.regime == 1:
            self._load_regime_1()
        if self.regime == 2:
            self._load_regime_2()

        return
    
    def _load_regime_1(self):
        """
        All inputs are correlated (correlation tuned with rho)
        """

        sigma = (1 - self.rho) * np.eye(self.n_X) + self.rho * np.ones([self.n_X, self.n_X])
        L = np.linalg.cholesky(sigma)
        Z = np.random.normal(loc=0, scale=1, size=[self.n_trials, self.n_X])
        X = Z @ L.T
        W = np.full(self.n_X, 1 / self.n_X).reshape(-1, 1)
        y = X @ W
        y = y + np.random.normal(loc=0, scale=self.eps, size=len(y)).reshape(-1, 1)
        X_noise = np.random.normal(loc=0, scale=self.eps, size=X.shape)
        X = X + X_noise
        self._inputs = X # Kinematic features
        self._output = y[..., None] # Neural activity

        return
    
    def _load_regime_2(self):
        """
        All inputs but one are correlated and the uncorrelated input is used to generate y
        """

        X_0 = np.random.normal(loc=0, scale=1, size=[self.n_trials, 1])
        W = np.ones(1).reshape(-1, 1)
        y = X_0 @ W
        y = y + np.random.normal(loc=0, scale=self.eps, size=len(y)).reshape(-1, 1)
        sigma = (1 - self.rho) * np.eye(self.n_X - 1) + self.rho * np.ones([self.n_X - 1, self.n_X - 1])
        L = np.linalg.cholesky(sigma)
        Z = np.random.normal(loc=0, scale=1, size=[self.n_trials, self.n_X - 1])
        E = Z @ L.T

        # Weighted sum of signal and nuisance variables
        X_nuisance = (
            self.cor * (X_0 @ np.ones((1, self.n_X - 1)))
            + np.sqrt(1 - self.cor**2) * E
        )
        X = np.hstack([
            X_0,
            X_nuisance
        ])
        X_noise = np.random.normal(loc=0, scale=self.eps, size=X.shape)
        X = X + X_noise
        self._inputs = X # Kinematic features
        self._output = y[..., None] # Neural activity

        return
    
    def standardize_X(self, X=None):
        return self.output
    
    def decompose_X(self, X=None, n_components=None):
        return self.output
    
    def filter_X(self, unit_types):
        return self.output
    
    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output
    
    @property
    def X(self):
        return self._output
    
    @property
    def saccade_direction(self):
        return self.inputs[:, 0]
    
    @property
    def saccade_amplitude(self):
        return self.inputs[:, 1]
    
    @property
    def saccade_startpoints(self):
        return self.inputs[:, 2]
    
    @property
    def saccade_endpoints(self):
        return self.inputs[:, 3]