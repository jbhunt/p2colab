import numpy as np
import torch
from .utils import MlatiSessionDataset
from .models import Seq2SeqDecoder
from sklearn.linear_model import LinearRegression
    
class Result():

    """
    """

    _statistic = None
    _null = None
    _d = None
    _p = None

    def __init__(self, statistic, null, baseline=None):
        """
        """

        self._statistic = round(statistic, 3).item()
        self._null = null
        self._p = ((np.sum(null <= statistic) + 1) / (len(null) + 1)).item()
        self._d = ((null.mean() - statistic) / null.std()).item()

        if baseline is None:
            self._r = None
        else:
            self._r = round(1 - (statistic / baseline), 3)

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
    def r(self):
        return self._r
    
class Residualizer():
    """
    """

    def __init__(self):
        self.reg = LinearRegression() # TODO: Allow residualizer to use logistic regression in the case of a classificaiton task
        return
    
    def fit(self, ds, target_feature_name, nuisance_feature_names):
        """
        """

        nuisance_features = list()
        for name in nuisance_feature_names:
            nuisance_feature = getattr(ds, name)
            nuisance_features.append(nuisance_feature)
        nuisance_features = np.array(nuisance_features).T
        target_feature = getattr(ds, target_feature_name)
        self.reg.fit(nuisance_features, target_feature)

        return
    
    def predict(self, ds, target_feature_name, nuisance_feature_names):
        """
        """

        nuisance_features = list()
        for name in nuisance_feature_names:
            nuisance_feature = getattr(ds, name)
            nuisance_features.append(nuisance_feature)
        nuisance_features = np.array(nuisance_features).T
        target_feature = getattr(ds, target_feature_name)
        residuals = target_feature - self.reg.predict(nuisance_features)

        return residuals
    
class ConditionalDecodingExperiment():
    """
    """

    def __init__(self, src, ds=None):
        """
        """

        if ds is None:
            self.ds = MlatiSessionDataset(src)
            self.ds.load()
            self.ds.compress() # TODO: Make these transforms specific to the train split (i.e., stop test leakage)
            self.ds.standardize()
        else:
            self.ds = ds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.est = None

        return
    
    def run(
        self,
        n_runs=3,
        train_size=0.8,
        validation_size=0.1,
        lr=0.0001,
        max_iter=30,
        batch_size=32,
        split_seed=42
        ):
        """
        """

        #
        _, _, U = self.ds.X.shape
        self.est = Seq2SeqDecoder(
            input_size=U,
            output_size=1,
            lr=lr,
            max_iter=max_iter,
            batch_size=batch_size
        )

        #
        target_feature_names = (
            "saccade_amplitude",
            "saccade_endpoints",
            "saccade_startpoints"
        )

        #
        results = {}
        n_jobs = len(target_feature_names) * (n_runs + 2)
        i_job = 0
        for target_feature_name in target_feature_names:

            #
            print(f"Working on job {i_job + 1} out of {n_jobs}")

            # Create fresh subsets
            ds_train, ds_test = self.ds.random_split([train_size, 1 - train_size], split_seed=split_seed)
            ds_train, ds_valid = ds_train.random_split([1 - validation_size, validation_size], split_seed=split_seed)

            # Perform a baseline fit
            for ds in (ds_train, ds_valid, ds_test):
                y_baseline = getattr(ds, target_feature_name)
                ds.clear_overrides()
                ds.set_y(y_baseline)
            self.est._return_to_initial_state()
            self.est.fit(ds_train, ds_valid, print_info=False)
            rmse_baseline = self.est.score(ds_test)

            #
            nuisance_feature_names = [n for n in target_feature_names if n != target_feature_name]

            # Residualize targets
            res = Residualizer()
            res.fit(ds_train, target_feature_name, nuisance_feature_names)
            for ds in (ds_train, ds_valid, ds_test):
                y_residual = res.predict(ds, target_feature_name, nuisance_feature_names)  
                ds.clear_overrides()
                ds.set_y(y_residual)

            # Fit and score
            self.est._return_to_initial_state()
            self.est.fit(ds_train, ds_valid, print_info=False)
            rmse_test = self.est.score(ds_test)

            #
            i_job += 1
            
            # Run permutation test
            null = np.full(n_runs, np.nan)
            y_train_unshuffled = np.copy(ds_train.y)
            y_valid_unshuffled = np.copy(ds_valid.y)
            for i_run in range(n_runs):

                #
                print(f"Working on job {i_job + 1} out of {n_jobs}")

                # Shuffle trials for training dataset
                y_train_shuffled = np.copy(y_train_unshuffled)
                shuffled_index = np.random.choice(np.arange(len(y_train_shuffled)), size=len(y_train_shuffled), replace=False)  
                y_train_shuffled = y_train_shuffled[shuffled_index]
                ds_train.set_y(y_train_shuffled)

                # Shuffle trials for validation dataset
                y_valid_shuffled = np.copy(y_valid_unshuffled)
                shuffled_index = np.random.choice(np.arange(len(y_valid_shuffled)), size=len(y_valid_shuffled), replace=False)  
                y_valid_shuffled = y_valid_shuffled[shuffled_index]
                ds_valid.set_y(y_valid_shuffled)

                # Refit
                self.est._return_to_initial_state()   
                self.est.fit(ds_train, ds_valid, print_info=False)

                # Eval
                null[i_run] = self.est.score(ds_test)

                # Reset target (just for hygiene)
                ds_train.set_y(y_train_unshuffled)
                ds_valid.set_y(y_valid_unshuffled)

                #
                i_job += 1

            #
            res = Result(rmse_test, null, baseline=rmse_baseline)

            #
            results[target_feature_name] = res          

        return results
    