import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, ShuffleSplit
import polars as pl
from pathlib import Path
from datetime import date
import copy

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
        self._saccade_velocity = None
        self._saccade_blocks = None
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
            grating_onset = np.array(stream["stimuli/dg/grating/timestamps"])
            grating_offset = np.array(stream["stimuli/dg/iti/timestamps"])
            # grating_motion = np.array(stream["stimuli/dg/grating/motion"])

        # For some experiments there are different numbers of frames and timestamps which will preclude further processing
        if frame_timestamps.size != eye_position.size:
            raise Exception(f'Different number of frames ({eye_position.size}) and frame timestamps ({frame_timestamps.size})')
        
        # Drop saccades without timestamps
        invalid_saccades = np.isnan(saccade_timestamps).any(1)
        saccade_labels = np.delete(saccade_labels, invalid_saccades)
        saccade_timestamps = np.delete(saccade_timestamps, invalid_saccades, axis=0)
        saccade_onset_timestamps = saccade_timestamps[:, 0]
        saccade_offset_timestamps = saccade_timestamps[:,1]

        # Code block
        saccade_blocks = list()
        for t2 in saccade_onset_timestamps:
            block_found = False
            for block, (t1, t3) in enumerate(zip(grating_onset, grating_offset)):
                if t1 <= t2 < t3:
                    block_found = True
                    break
            if block_found == False:
                block = -1
            saccade_blocks.append(block)
        saccade_blocks = np.array(saccade_blocks)

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
        saccade_velocity = list()
        t_eval = y_binsize * (np.arange(-1 * y_bincounts[0], y_bincounts[1], 1) + 0.5)
        self._t_y = t_eval
        for event_timestamp in event_timestamps:
            wf = np.interp(
                t_eval + event_timestamp,
                t_raw,
                y_raw
            )
            peak_velocity = np.max(np.abs(np.diff(wf)))
            saccade_velocity.append(peak_velocity)

            # Need to scale signal if derivative = 1
            if derivative == 1: # TODO: Check if this is right
                wf =  wf / y_binsize
            saccade_waveforms.append(wf)
        saccade_waveforms = np.array(saccade_waveforms)

        #
        saccade_startpoints = list()
        saccade_endpoints = list()
        # saccade_velocity = list()
        for t1, t2 in zip(saccade_onset_timestamps, saccade_offset_timestamps):
            p1 = np.interp(t1, t_raw, y_raw)
            p2 = np.interp(t2, t_raw, y_raw)
            saccade_startpoints.append(p1)
            saccade_endpoints.append(p2)
            # v = abs(p2 - p1) / (t2 - t1) # Displacement over duration
            # saccade_velocity.append(v)
        saccade_startpoints = np.array(saccade_startpoints)
        saccade_endpoints = np.array(saccade_endpoints)
        saccade_amplitude = np.abs(saccade_endpoints - saccade_startpoints)
        saccade_velocity = np.array(saccade_velocity)

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
        all_edges = np.concatenate([left_edges, [right_edges[-1],]]) * X_binsize
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
        R = np.array(R, dtype=np.float16) # U x N x T
        X = np.swapaxes(R, 0, 1) # N x U x T
        self._X = np.swapaxes(X, 1, 2) # N x T x U
        self._X_raw = np.copy(self._X)
        self._y = saccade_waveforms # Default target is saccade waveforms (but this can be overriden with the set_y method)
        self._y_raw = np.copy(self._y)
        self._saccade_waveforms = saccade_waveforms
        self._saccade_startpoints = saccade_startpoints
        self._saccade_endpoints = saccade_endpoints
        self._saccade_amplitude = saccade_amplitude
        self._saccade_velocity = saccade_velocity
        self._saccade_direction = z
        self._saccade_blocks = saccade_blocks

        #
        self._loaded = True

        return
    
    def filter_X(self, unit_types=("premotor", "visuomotor", "visual")):
        """
        Return a view of X for a subset of unit types
        """

        if unit_types is None:
            return self.X

        if self.lut is None:
            raise Exception("Lookup table not specificed at instantiation")
        
        lut = pl.read_csv(self.lut).with_columns(
            pl.col("date").str.to_date("%Y-%m-%d")
        )
        lut = lut.filter((pl.col("date") == self.date) & (pl.col("animal") == self.animal) & (pl.col("utype").is_in(unit_types)))
        target_unit_ids = lut.select("uid").to_numpy().flatten()
        _, index, _ = np.intersect1d(self.unit_ids, target_unit_ids, return_indices=True)
        X_out = self.X[:, :, index]

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
        self._X = self._X_raw

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
        self._y = self._y_raw

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
        ds.date = self.date
        ds.animal = self.animal
        ds._unit_ids = self._unit_ids

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
            ds._saccade_velocity = self._saccade_velocity[new_index].copy()
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
            ds._saccade_velocity = self._saccade_velocity
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
    
    def kfold_split(self, k=5, validation_fraction=0.1, split_seed=42):
        """
        """

        cv = KFold(n_splits=k, shuffle=True, random_state=split_seed)
        splits = {
            "train": list(),
            "test": list(),
            "valid": list()
        }
        for i_split, (mixed_indices, test_indices) in enumerate(cv.split(self.X)):
            if validation_fraction is not None:
                ss = ShuffleSplit(
                    n_splits=1,
                    test_size=validation_fraction,
                    random_state=split_seed + i_split, # Reproducible per fold
                )
                train_indices_, valid_indices_ = next(ss.split(mixed_indices))
                train_indices = mixed_indices[train_indices_]
                valid_indices = mixed_indices[valid_indices_]
            else:
                train_indices = mixed_indices
                valid_indices = None
            ds_train = self.spawn_subset(train_indices, copy=True)
            ds_test = self.spawn_subset(test_indices, copy=True)
            if validation_fraction is not None:
                ds_valid = self.spawn_subset(valid_indices, copy=True)
            splits["train"].append(ds_train)
            splits["test"].append(ds_test)
            splits["valid"].append(ds_valid)

        return splits
    
    @property
    def loaded(self):
        return self._loaded
    
    @property
    def t_X(self):
        return self._t_X
    
    @property
    def t_y(self):
        return self._t_y
    
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
    def saccade_velocity(self):
        return self._apply_index(self._overrides.get("saccade_startpoints", self._saccade_velocity))
    
    @property
    def saccade_blocks(self):
        return self._apply_index(self._overrides.get("saccade_blocks", self._saccade_blocks))
    
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
    
class PsuedoSessionDataset(Dataset):
    """
    """

    def __init__(
        self,
        sessions,
        n_trials=None,
        n_levels=100,
        feature_range=(-5, 5),
        max_dist=0.5,
        k=5,
        build=True,
        random_seed=42
        ):
        """
        """

        super().__init__()
        self.sessions = [] if sessions is None else sessions
        self._X = None
        self._X_raw = None
        self._y = None
        self._y_raw = None
        self._X_pseudo = None
        self._y_pseudo = None
        self._saccade_direction = None
        self._saccade_amplitude = None
        self._saccade_startpoints = None
        self._saccade_endpoints = None
        self._saccade_velocity = None
        self._saccade_blocks = None
        self._t_X = None
        self._anchor_session_index = None
        self._session_unit_slices = None
        self.n_levels = n_levels
        self.feature_range = feature_range
        self.n_trials = n_trials
        self.max_dist = max_dist
        if self.max_dist is None:
            self.max_dist = np.inf
        self.k = k
        self.random_seed = random_seed
        self.session_keys = [(s.animal, str(s.date)) for s in self.sessions]

        if build and len(self.sessions) > 0:
            self.build()

        return
    
    def _process_sessions(self):
        """
        Standardize and digitize kinematic features
        """

        attrs = (
            "saccade_direction",
            "saccade_amplitude",
            "saccade_startpoints",
            "saccade_endpoints",
            "saccade_velocity",
            "saccade_blocks",
        )
        bin_edges = np.linspace(*self.feature_range, self.n_levels + 1)
        bin_centers = np.vstack([
            bin_edges[: -1],
            bin_edges[1:]
        ]).mean(0)
        self._trial_data = dict()
        block_offset = 0
        for s in self.sessions:
            ys = []

            # normalize block labels to start at 0 within session, then offset
            saccade_blocks = np.asarray(s.saccade_blocks)
            saccade_blocks = saccade_blocks - saccade_blocks.min()

            for a in attrs:
                if a == "saccade_direction":
                    y = np.asarray(getattr(s, a))

                elif a == "saccade_blocks":
                    y = (saccade_blocks + block_offset)

                else:
                    sample = np.asarray(getattr(s, a))
                    xbar = sample.mean(axis=0, keepdims=True)
                    sd = sample.std(axis=0, keepdims=True)
                    sd = np.where(sd == 0, 1, sd)
                    normed = (sample - xbar) / sd
                    normed = np.clip(normed, *self.feature_range)
                    bin_indices = np.digitize(normed, bin_edges, right=True) - 1
                    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
                    y = bin_centers[bin_indices]

                ys.append(y)

            trial_data = np.vstack(ys).T
            self._trial_data[(s.animal, str(s.date))] = trial_data
            block_offset += int(np.max(saccade_blocks)) + 1

        #
        self._session_unit_slices = []
        c0 = 0
        for s in self.sessions:
            c1 = c0 + s.X.shape[-1]
            self._session_unit_slices.append(slice(c0, c1))
            c0 = c1

        return
    
    def _build_pseudotrials(self):
        """
        Build pseudotrials
        - Randomly select an anchor session and trial
        - Pull the most similar trial from all other sessions
        - Consume anchors and candidates only if a full pseudotrial is successfully built
        - Concatenate neural activity
        """

        X_pseudo = []
        y_pseudo = []
        anchor_session_indices = []
        provenance = []

        rng = np.random.default_rng(self.random_seed)

        # Unused anchor trials, grouped by session
        anchor_pool = {
            i_session: list(range(s.X.shape[0]))
            for i_session, s in enumerate(self.sessions)
        }

        # Unused candidate trials, grouped by session
        candidate_pool = {
            i_session: list(range(s.X.shape[0]))
            for i_session, s in enumerate(self.sessions)
        }

        while True:
            valid_anchor_sessions = [
                i_session
                for i_session, trials in anchor_pool.items()
                if len(trials) > 0
            ]            
            if len(valid_anchor_sessions) == 0:
                break

            # Pick an anchor session/trial, but do not consume yet
            i_anchor = rng.choice(valid_anchor_sessions)
            anchor_trial_pos = rng.integers(len(anchor_pool[i_anchor]))
            i_trial = anchor_pool[i_anchor][anchor_trial_pos]

            anchor_session = self.sessions[i_anchor]
            anchor_key = self.session_keys[i_anchor]
            anchor_trial = self._trial_data[anchor_key][i_trial]
            saccade_direction = anchor_trial[0]

            pieces = [anchor_session.X[i_trial]]
            selected_candidates = []
            pseudo_members = [(i_anchor, i_trial)]
            failed = False

            for i_candidate, s in enumerate(self.sessions):
                if i_candidate == i_anchor:
                    continue

                available_trials = candidate_pool[i_candidate]
                if len(available_trials) == 0:
                    failed = True
                    break

                candidate_key = self.session_keys[i_candidate]
                candidate_trials = self._trial_data[candidate_key][available_trials]

                # Restrict to same direction
                mask = candidate_trials[:, 0] == saccade_direction
                if not np.any(mask):
                    failed = True
                    break

                # Match on amplitude, startpoint, endpoint, velocity
                dists = np.linalg.norm(
                    candidate_trials[:, 1:-1] - anchor_trial[1:-1],
                    axis=1
                )
                
                dists[~mask] = np.nan

                finite = np.isfinite(dists)
                if not np.any(finite):
                    failed = True
                    break

                min_dist = np.nanmin(dists)
                if min_dist > self.max_dist:
                    failed = True
                    break

                valid_local = np.where(np.isfinite(dists) & (dists <= self.max_dist))[0]
                if len(valid_local) == 0:
                    failed = True
                    break

                order = np.argsort(dists[valid_local])
                valid_local = valid_local[order]
                k_eff = min(self.k, len(valid_local))
                chosen_local = rng.choice(valid_local[:k_eff])
                best_match = available_trials[chosen_local]

                selected_candidates.append((i_candidate, best_match))
                pieces.append(s.X[best_match])
                pseudo_members.append((i_candidate, best_match))

            if failed:
                # This anchor could not be completed. Remove it from future anchor attempts,
                # but do not consume any candidates.
                anchor_pool[i_anchor].pop(anchor_trial_pos)
                continue

            # Commit consumption only after full success
            anchor_pool[i_anchor].pop(anchor_trial_pos)

            # Prevent this real trial from ever being used later as an anchor or candidate
            if i_trial in candidate_pool[i_anchor]:
                candidate_pool[i_anchor].remove(i_trial)

            for i_candidate, best_match in selected_candidates:
                candidate_pool[i_candidate].remove(best_match)
                if best_match in anchor_pool[i_candidate]:
                    anchor_pool[i_candidate].remove(best_match)

            x_trial = np.concatenate(pieces, axis=-1)
            X_pseudo.append(x_trial)
            y_pseudo.append(anchor_trial)
            anchor_session_indices.append(i_anchor)
            provenance.append(pseudo_members)

            if self.n_trials is not None and len(X_pseudo) >= self.n_trials:
                break

        if len(X_pseudo) == 0:
            raise ValueError("Pseudo-trial construction failed")

        n_trials = len(X_pseudo)

        self._anchor_session_index = None
        self._anchor_session_indices = np.asarray(anchor_session_indices, dtype=int)
        self._provenance = provenance
        self._X_pseudo = np.asarray(X_pseudo)
        self._X = self._X_pseudo.copy()
        self._X_raw = self._X_pseudo.copy()
        self._y_pseudo = np.asarray(y_pseudo)

        self._saccade_direction = self._y_pseudo[:, 0]
        self._saccade_amplitude = self._y_pseudo[:, 1]
        self._saccade_startpoints = self._y_pseudo[:, 2]
        self._saccade_endpoints = self._y_pseudo[:, 3]
        self._saccade_velocity = self._y_pseudo[:, 4]
        self._saccade_blocks = np.zeros(n_trials)

        self._y = self._saccade_direction.copy()
        self._y_raw = self._y.copy()
        self._t_X = self.sessions[0].t_X.copy()

        return
    
    def build(self):
        """
        """

        self._process_sessions()
        self._build_pseudotrials()

        return
    
    def reseed(self, random_seed):
        """
        """

        self.random_seed = random_seed
        self.build()

        return

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
    
    @property
    def X_pseudo(self):
        return self._X_pseudo
    
    @property
    def y_pseudo(self):
        return self._y_pseudo

    @property
    def t_X(self):
        return self._t_X

    @property
    def saccade_direction(self):
        return self._saccade_direction

    @property
    def saccade_amplitude(self):
        return self._saccade_amplitude

    @property
    def saccade_startpoints(self):
        return self._saccade_startpoints

    @property
    def saccade_endpoints(self):
        return self._saccade_endpoints
    
    @property
    def saccade_velocity(self):
        return self._saccade_velocity

    @property
    def saccade_blocks(self):
        return self._saccade_blocks

    def set_X(self, X):
        self._X = X
        return

    def reset_X(self):
        self._X = self._X_raw.copy()
        return

    def set_y(self, y):
        self._y = y
        return

    def reset_y(self):
        self._y = self._y_raw.copy()
        return

    def reset_Xy(self):
        self.reset_X()
        self.reset_y()
        return

    def filter_X(self, unit_types=("visual", "visuomotor", "premotor")):
        """
        """

        if unit_types is None:
            return self.X

        try:
            for s in self.sessions:
                s.reset_X()
                s.set_X(s.filter_X(unit_types=unit_types))
            self.build()

        finally:
            for s in self.sessions:
                s.reset_X()

        return self.X

    def spawn_subset(self, trial_indices, copy=True):
        """
        """

        ds = self.__class__.__new__(self.__class__)
        Dataset.__init__(ds)
        ds.sessions = self.sessions
        ds.session_keys = list(self.session_keys)
        ds._trial_data = self._trial_data
        ds._anchor_session_index = self._anchor_session_index
        ds.n_levels = self.n_levels
        ds._t_X = self._t_X.copy()
        ds._session_unit_slices = list(self._session_unit_slices)
        ds.feature_range = self.feature_range
        ds.random_seed = self.random_seed
        ds.n_trials = self.n_trials
        ds.max_dist = self.max_dist
        ds.k = self.k
        trial_indices = np.asarray(trial_indices)

        if copy:
            ds._X = self._X[trial_indices].copy()
            ds._X_raw = self._X_raw[trial_indices].copy()
            ds._y = self._y[trial_indices].copy()
            ds._y_raw = self._y_raw[trial_indices].copy()
            ds._X_pseudo = self._X_pseudo[trial_indices].copy()
            ds._y_pseudo = self._y_pseudo[trial_indices].copy()
            ds._anchor_session_indices = self._anchor_session_indices[trial_indices].copy()
            ds._saccade_direction = self._saccade_direction[trial_indices].copy()
            ds._saccade_amplitude = self._saccade_amplitude[trial_indices].copy()
            ds._saccade_startpoints = self._saccade_startpoints[trial_indices].copy()
            ds._saccade_endpoints = self._saccade_endpoints[trial_indices].copy()
            ds._saccade_velocity = self._saccade_velocity[trial_indices].copy()
            ds._saccade_blocks = self._saccade_blocks[trial_indices].copy()
        else:
            raise NotImplementedError("copy=False not implemented for PsuedoSessionDataset")

        return ds

    def kfold_split(self, k=5, validation_fraction=0.1, split_seed=42):
        """
        """

        cv = KFold(n_splits=k, shuffle=True, random_state=split_seed)
        splits = {
            "train": [],
            "test": [],
            "valid": [],
        }

        for i_split, (mixed_indices, test_indices) in enumerate(cv.split(self.X)):
            if validation_fraction is not None:
                ss = ShuffleSplit(
                    n_splits=1,
                    test_size=validation_fraction,
                    random_state=split_seed + i_split,
                )
                train_indices_, valid_indices_ = next(ss.split(mixed_indices))
                train_indices = mixed_indices[train_indices_]
                valid_indices = mixed_indices[valid_indices_]
            else:
                train_indices = mixed_indices
                valid_indices = None

            ds_train = self.spawn_subset(train_indices, copy=True)
            ds_test = self.spawn_subset(test_indices, copy=True)
            ds_valid = self.spawn_subset(valid_indices, copy=True) if validation_fraction is not None else None

            splits["train"].append(ds_train)
            splits["test"].append(ds_test)
            splits["valid"].append(ds_valid)

        return splits

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, index):
        X_i = self.X[index]
        y_i = self.y[index]
        return X_i, y_i
    
class SyntheticMlatiDataset(Dataset):
    """
    """

    def __init__(self, n_trials=1, regime=1, rho_within=0.7, rho_between=0.0, eps_signal=1, eps_noise_X=0.0, eps_noise_y=0.25, n_X=4):
        """
        inputs
        ------
        n_trials
            Number of simulated trials
        regime
            Experiment regime
        rho
            Correlation between nuisance variables
        cor
            Correlation between signal and nuisance variables
        eps
            Scale of noise
        """

        self.n_trials = n_trials
        self.n_X = n_X
        self.rho_between = rho_between
        self.rho_within = rho_within
        self.eps_signal = eps_signal
        self.eps_noise_X = eps_noise_X
        self.eps_noise_y = eps_noise_y
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
        All inputs are correlated (clones + iid noise) and are used to derive the output
        """

        X_0 = np.random.normal(loc=0, scale=self.eps_signal, size=[self.n_trials, 1])
        X = np.repeat(X_0, self.n_X)
        X = X + np.random.normal(loc=0, scale=self.eps_noise_X, size=X.shape)
        W = np.full(self.n_X, 1 / self.n_X).reshape(-1, 1)
        y = X @ W
        y = y + np.random.normal(loc=0, scale=self.eps_noise_y, size=len(y)).reshape(-1, 1)
        self._inputs = X # Kinematic features
        self._output = y[..., None] # Neural activity

        return
    
    def _load_regime_x(self):
        """
        """

        X_0 = np.random.normal(loc=0, scale=self.eps_signal, size=[self.n_trials, 1])
        X_1 = np.random.normal(loc=0, scale=self.eps_signal, size=[self.n_trials, 1])
        X_nuisance = [(np.copy(X_1) + np.random.normal(loc=0, scale=self.eps_noise_X)).reshape(-1, 1) for _ in range(self.n_X - 1)]
        X = np.hstack([
            X_0,
            *X_nuisance
        ])
        W = np.array([1, *np.zeros(self.n_X - 1)]).reshape(-1, 1)
        y = X @ W
        y = y + np.random.normal(loc=0, scale=self.eps_noise_y, size=len(y)).reshape(-1, 1)
        self._inputs = X # Kinematic features
        self._output = y[..., None] # Neural activity

        return
    
    def _load_regime_2(self):
        """
        """

        # 
        n = self.n_trials
        p = self.n_X - 1 # Number of nuisance variables
        rb = self.rho_between
        rw = self.rho_within

        # Determine ceiling for correlation between signal and nuisance variables
        if rw < rb ** 2:
            raise ValueError(
                f"Need rho_within >= rho_between^2. Got {rw} < {rb ** 2}"
            )

        # Latent variables
        S = np.random.normal(size=(n, 1), scale=self.eps_signal) # Signal factor
        U = np.random.normal(size=(n, 1), scale=self.eps_signal) # Nuisance factor (shared)
        E = np.random.normal(size=(n, p), scale=self.eps_signal) # Independent noise

        # Compute coefficients
        a = rb
        b = np.sqrt(rw - rb ** 2)
        c = np.sqrt(1 - a ** 2 - b ** 2)

        # Generate signal and nuisance variables
        X_0 = S
        X_nuisance = a * S + b * U + c * E
        X = np.hstack([X_0, X_nuisance])

        # Create target
        # NOTE: Important to do this before adding noise
        y = X_0
        
        # Add noise to inputs and outputs
        y = y + np.random.normal(scale=self.eps_noise_y, size=(n, 1))
        X = X + np.random.normal(scale=self.eps_noise_X, size=X.shape)

        #
        self._inputs = X
        self._output = y[..., None]

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