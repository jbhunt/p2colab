import numpy as np
import torch
from torch.utils.data import Dataset

def raised_cosine_phi(T, center=0.0, width=0.35):
    t = np.linspace(-1, 1, T).astype(np.float32)
    x = (t - center) / (width + 1e-8)
    phi = np.zeros_like(t)
    m = np.abs(x) <= 1
    phi[m] = 0.5 * (1.0 + np.cos(np.pi * x[m]))
    phi = phi / (np.linalg.norm(phi) + 1e-8)
    return phi

class SimpleMlatiLikeDataset(Dataset):
    """
    Mimics what your LeaveOneOutEncodingExperiment expects:
      - ds.X is neural (N,T,U)
      - ds.saccade_* are behavioral vectors (N,)
      - supports set_X / reset_X because you overwrite X with behavioral design matrix
    """
    def __init__(self, X_neural, beh_mat, feature_names=None):
        super().__init__()
        self._X_raw = np.asarray(X_neural).astype(np.float32)
        self._X = self._X_raw.copy()
        self._y = None  # your experiment will set this

        if feature_names is None:
            feature_names = ["saccade_direction","saccade_amplitude","saccade_startpoints","saccade_endpoints"]
        self.feature_names = list(feature_names)
        self._name_to_j = {n:j for j,n in enumerate(self.feature_names)}
        self._beh = np.asarray(beh_mat).astype(np.float32)  # (N,F)

    # neural input (starts as N,T,U, later overwritten by behavioral design matrix N,F)
    @property
    def X(self): return self._X

    @property
    def y(self): return self._y

    def set_X(self, X): self._X = np.asarray(X).astype(np.float32)
    def reset_X(self): self._X = self._X_raw.copy()
    def set_y(self, y): self._y = np.asarray(y).astype(np.float32)
    def reset_y(self): self._y = None
    def reset_Xy(self): self.reset_X(); self.reset_y()

    # behavioral accessors
    @property
    def saccade_direction(self): return self._beh[:, self._name_to_j["saccade_direction"]]
    @property
    def saccade_amplitude(self): return self._beh[:, self._name_to_j["saccade_amplitude"]]
    @property
    def saccade_startpoints(self): return self._beh[:, self._name_to_j["saccade_startpoints"]]
    @property
    def saccade_endpoints(self): return self._beh[:, self._name_to_j["saccade_endpoints"]]

    def __len__(self): return self._X.shape[0]

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]

    def spawn_subset(self, indices):
        indices = np.asarray(indices)
        ds = SimpleMlatiLikeDataset(
            X_neural=self._X_raw[indices].copy(),
            beh_mat=self._beh[indices].copy(),
            feature_names=self.feature_names
        )
        # if y already set, subset it too
        if self._y is not None:
            ds.set_y(self._y[indices].copy())
        return ds

    def random_split(self, split_sizes=(0.8,0.2), split_seed=42):
        if not np.isclose(sum(split_sizes), 1.0, rtol=1e-3):
            raise ValueError("Split sizes must sum to 1")
        n_total = self._X_raw.shape[0]
        counts = [int(round(s*n_total)) for s in split_sizes]
        counts[-1] = n_total - sum(counts[:-1])

        g = torch.Generator().manual_seed(int(split_seed))
        perm = torch.randperm(n_total, generator=g).cpu().numpy()

        splits = []
        i0 = 0
        for c in counts:
            idx = perm[i0:i0+c]
            splits.append(self.spawn_subset(idx))
            i0 += c
        return splits

def make_simple_encoding_dataset_for_your_experiment(
    N=600, T=50, U=40, seed=0, snr=8.0, noise_sd=1.0, drive_feature="saccade_amplitude"
):
    rng = np.random.default_rng(int(seed))

    feature_names = ["saccade_direction","saccade_amplitude","saccade_startpoints","saccade_endpoints"]
    F = len(feature_names)
    name_to_j = {n:i for i,n in enumerate(feature_names)}

    # behavioral features
    beh = rng.normal(size=(N, F)).astype(np.float32)

    # driving scalar
    j = name_to_j[drive_feature]
    x = beh[:, j]
    x = (x - x.mean()) / (x.std() + 1e-8)

    # shared timecourse + positive unit gains so signal is strong and easy
    phi = raised_cosine_phi(T, center=0.0, width=0.35)  # (T,)
    gains = rng.uniform(0.5, 1.5, size=U).astype(np.float32)

    signal = snr * x[:, None, None] * phi[None, :, None] * gains[None, None, :]  # (N,T,U)
    noise = rng.normal(scale=noise_sd, size=(N, T, U)).astype(np.float32)
    X_neural = signal + noise

    return SimpleMlatiLikeDataset(X_neural=X_neural, beh_mat=beh, feature_names=feature_names)
