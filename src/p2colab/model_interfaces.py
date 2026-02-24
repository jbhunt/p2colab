import torch
from .linear_models import RidgeWithTimeBasis
from .nonlinear_models import MLP, RNN
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from sklearn.linear_model import Ridge
from torch import nn
import copy

class Decoder():
    """
    """

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=5,
        C_out=32,
        dropout=0.1,
        lr=0.001,
        max_iter=30,
        batch_size=None,
        patience=15,
        task_type="regression"
        ):
        """
        """

        self.output_size = output_size
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.loss = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RNN(
            C_in=input_size,
            output_size=output_size,
            C_out=C_out,
            dropout=dropout,
            kernel_size=kernel_size
        )
        self._init_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.patience = patience
        self.task_type = task_type # TODO: Extend the model to handle classification tasks (for saccade direction)

        return
    
    def _return_to_initial_state(self):
        """
        """

        self.model.load_state_dict(self._init_state_dict)
        self.model.to(self.device)

        return
    
    def fit(self, ds_train, ds_valid, print_info=True):
        """
        """

        if self.batch_size is None:
            batch_size_train = len(ds_train)
            batch_size_valid = len(ds_valid)
        else:
            batch_size_train = batch_size_valid = self.batch_size
        ldr_train = DataLoader(ds_train, batch_size=batch_size_train, shuffle=True)
        ldr_valid = DataLoader(ds_valid, batch_size=batch_size_valid, shuffle=False)
        loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        #
        self.loss = np.full([self.max_iter, 2], np.nan)
        best_state_dict = None
        best_loss = np.inf
        i_step = 0
        n_epochs_no_improve = 0
        for i_epoch in range(self.max_iter):

            # Train
            batch_loss_train = 0
            self.model.train()
            for X_b, y_b in ldr_train:
                X_b = X_b.to(device=self.device, dtype=torch.float32)
                y_b = y_b.to(device=self.device, dtype=torch.float32)
                y_pred = self.model(X_b)
                if self.output_size == 1:
                    y_pred = y_pred.squeeze(-1)
                loss = loss_fn(y_pred, y_b)
                batch_loss_train += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i_step += 1
            batch_loss_train /= len(ldr_train)

            # Eval
            self.model.eval()
            batch_loss_valid = 0
            with torch.no_grad():
                for X_b, y_b in ldr_valid:
                    X_b = X_b.to(device=self.device, dtype=torch.float32)
                    y_b = y_b.to(device=self.device, dtype=torch.float32)
                    y_pred = self.model(X_b)
                    if self.output_size == 1:
                        y_pred = y_pred.squeeze(-1)
                    loss = loss_fn(y_pred, y_b)
                    batch_loss_valid += loss.item()
            batch_loss_valid /= len(ldr_valid)

            #
            if print_info:
                print(f"Epoch {i_epoch + 1}: train loss={batch_loss_train:.3f}, validation loss={batch_loss_valid:.3f}")
            self.loss[i_epoch, 0] = batch_loss_train
            self.loss[i_epoch, 1] = batch_loss_valid

            #
            if batch_loss_valid < best_loss:
                best_loss = batch_loss_valid
                # best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_state_dict = copy.deepcopy(self.model.state_dict())
                n_epochs_no_improve = 0
            else:
                n_epochs_no_improve += 1
            if n_epochs_no_improve >= self.patience:
                if print_info:
                    print("Early stopping triggered")
                break

        # Load the model from the best epoch
        best_state_dict = {k: v.detach().cpu().clone() for k, v in best_state_dict.items()}
        self.model.load_state_dict(best_state_dict)

        return
    
    def predict(self, ds):
        """
        """

        X, y = ds[:]
        X_ = torch.from_numpy(X).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_).detach().cpu().numpy()

        return y_pred
    
    def score(self, ds):
        """
        """

        X_true, y_true = ds[:]
        y_pred = self.predict(ds)
        rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

        return rmse
    
    def score_r2(self, ds):
        """
        """

        _, y_true = ds[:]
        y_pred = self.predict(ds)
        floor = np.sum(np.power(y_true - y_true.mean(), 2))
        residual = np.sum(np.power(y_true - y_pred, 2))
        r2 = 1 - (residual / floor)

        return r2

class LinearEncoder1():
    """
    Ridge encoder with time basis
    """

    def __init__(
        self,
        F,
        T,
        C=1,
        K=7,
        l2_penalty=0.001,
        width_scale=1.0,
        patience=15,
        lr=0.001,
        max_iter=30,
        batch_size=32,
        ):
        """
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.l2_penalty = l2_penalty
        self.K = K
        self.width_scale = width_scale
        self.patience = patience
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model = RidgeWithTimeBasis(input_size=F, T=T, C=C, K=self.K, width_scale=self.width_scale)
        self._init_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        return
    
    def _return_to_initial_state(self):
        """
        """

        self.model.load_state_dict(self._init_state_dict)

        return
    
    def fit(self, ds_train, ds_valid, print_info=True):
        """
        """

        #
        N, F = ds_train.X.shape
        N, T, C = ds_train.y.shape
        self.model.to(self.device)

        #
        ldr_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        ldr_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        #
        best_loss = np.inf
        best_state_dict = None
        n_epochs_no_improve = 0

        #
        for i_epoch in range(self.max_iter):

            #
            batch_loss_train = 0.0
            self.model.train()
            for X_b, y_b in ldr_train:
                X_b = X_b.to(self.device, dtype=torch.float32)
                y_b = y_b.to(self.device, dtype=torch.float32)
                if y_b.ndim == 2:
                    y_b = y_b.unsqueeze(-1)  # B x T x 1
                y_pred = self.model(X_b) # B x T x C
                mse = torch.mean((y_pred - y_b) ** 2)
                W = self.model.linear.weight
                loss = mse + self.l2_penalty * torch.mean(W[:, 1:] ** 2)
                batch_loss_train += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_loss_train /= len(ldr_train)

            #
            batch_loss_valid = 0.0
            self.model.eval()
            with torch.no_grad():
                for X_b, y_b in ldr_valid:
                    X_b = X_b.to(self.device, dtype=torch.float32)
                    y_b = y_b.to(self.device, dtype=torch.float32)
                    if y_b.ndim == 2:
                        y_b = y_b.unsqueeze(-1)  # B x T x 1
                    y_pred = self.model(X_b)     # B x T x C
                    mse = torch.mean((y_pred - y_b) ** 2)
                    W = self.model.linear.weight
                    loss = mse + self.l2_penalty * torch.mean(W[:, 1:] ** 2)
                    batch_loss_valid += loss.item()
            batch_loss_valid /= len(ldr_valid)

            #
            if print_info:
                print(f"Epoch {i_epoch + 1}: train loss={batch_loss_train:.6f}, validation loss={batch_loss_valid:.6f}")

            #
            if batch_loss_valid < best_loss:
                best_loss = batch_loss_valid
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                n_epochs_no_improve = 0
            else:
                n_epochs_no_improve += 1
            if n_epochs_no_improve >= self.patience:
                if print_info:
                    print("Early stopping triggered")
                break

        #
        self.model.load_state_dict(best_state_dict)

        return
    
    def predict(self, ds):
        """
        """

        X, y = ds[:]
        X_ = torch.from_numpy(X).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_).detach().cpu().numpy()

        return y_pred
    
    def score(self, ds):
        """
        """

        X_true, y_true = ds[:]
        y_pred = self.predict(ds)
        rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2))).item()

        return rmse
    
    def score_r2(self, ds):
        """
        """

        X_true, y_true = ds[:]
        y_pred = self.predict(ds)
        floor = np.sum(np.power(y_true - y_true.mean(), 2))
        residual = np.sum(np.power(y_true - y_pred, 2))
        r2 = 1 - (residual / floor)

        return r2
    
class LinearEncoder2:
    """
    Ridge encoder with time basis
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None
        self.T_ = None
        self.C_ = None
        self.D_ = None

    def _return_to_initial_state(self): return

    def fit(self, ds_train, ds_valid, print_info=False):
        X, y = ds_train[:]
        X = np.asarray(X, float)   # (N, D)
        Y = np.asarray(y, float)   # (N, T, C)

        N, D = X.shape
        N2, T, C = Y.shape
        assert N == N2

        self.T_ = T
        self.C_ = C
        self.D_ = D

        # Time basis: identity
        B = np.eye(T)                     # (T, T)
        B_big = np.tile(B, (N, 1))        # (N*T, T)

        # Feature interactions with time
        X_rep = np.repeat(X, T, axis=0)   # (N*T, D)
        X_time = np.concatenate(
            [B_big] + [B_big * X_rep[:, d:d+1] for d in range(D)],
            axis=1
        )                                 # (N*T, T*(D+1))

        Y_flat = Y.reshape(N*T, C)

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_time, Y_flat)
        return self

    def predict(self, ds):
        X, _ = ds[:]
        X = np.asarray(X, float)
        N, D = X.shape
        assert D == self.D_

        T = self.T_
        C = self.C_

        B = np.eye(T)
        B_big = np.tile(B, (N, 1))
        X_rep = np.repeat(X, T, axis=0)

        X_time = np.concatenate(
            [B_big] + [B_big * X_rep[:, d:d+1] for d in range(D)],
            axis=1
        )

        Y_flat = self.model.predict(X_time)
        return Y_flat.reshape(N, T, C)
    
    def score_r2(self, ds):
        """
        """

        _, y_true = ds[:]
        y_pred = self.predict(ds)
        floor = np.sum(np.power(y_true - y_true.mean(), 2))
        residual = np.sum(np.power(y_true - y_pred, 2))
        r2 = 1 - (residual / floor)

        return r2
    
class LinearEncoder3():
    """
    Simplest linear encoder
    """

    def __init__(self, alpha=1.0):
        """
        """

        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

        return
    
    def _return_to_initial_state(self): return
    
    def fit(self, ds_train, ds_valid=None, print_info=False):
        """
        """

        X, y = ds_train[:]
        N, T, C = y.shape
        y_flat = y.reshape(-1, T * C)
        self.model.fit(X, y_flat)

        return self
    
    def predict(self, ds):
        """
        """

        X, y = ds[:]
        N, T, C = y.shape
        y_pred = self.model.predict(X).reshape(N, T, C)

        return y_pred
    
    def score_r2(self, ds):
        """
        """

        _, y_true = ds[:]
        y_pred = self.predict(ds)
        rss = np.sum(np.power(y_true - y_pred, 2))
        tss = np.sum(np.power(y_true - y_true.mean(), 2))
        r2 = round(float(1 - (rss / tss)), 6)

        return r2
    
class NonlinearEncoder:
    """
    Behavior/features X (N,F) -> Neural target y (N,T) or (N,T,C) using an MLP.

    The MLP predicts a flattened vector of size (T*C) and we reshape to (T,C).
    """

    def __init__(
        self,
        F: int,
        T: int,
        C: int = 1,
        hidden_layer_sizes=(32,),
        dropout: float = 0.1,
        weight_decay: float = 0.0,   # AdamW weight decay (L2)
        patience: int = 20,
        lr: float = 1e-3,
        max_iter: int = 30,
        batch_size: int = 32,
        grad_clip_norm: float | None = None,
        ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.F = int(F)
        self.T = int(T)
        self.C = int(C)

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)

        self.patience = int(patience)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.grad_clip_norm = grad_clip_norm

        self.model = MLP(
            input_size=self.F,
            output_size=self.T * self.C,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=self.dropout,
        )

        # Save init weights for reset between runs
        self._init_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _return_to_initial_state(self):
        self.model.load_state_dict(self._init_state_dict)

    def _ensure_y_3d(self, y_b: torch.Tensor) -> torch.Tensor:
        # Accept (B,T) or (B,T,C); return (B,T,C)
        if y_b.ndim == 2:
            return y_b.unsqueeze(-1)
        if y_b.ndim == 3:
            return y_b
        raise ValueError(f"Expected y with 2 or 3 dims, got shape {tuple(y_b.shape)}")

    def fit(self, ds_train, ds_valid, print_info: bool = True):
        self.model.to(self.device)

        ldr_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        ldr_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss = np.inf
        best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        n_epochs_no_improve = 0

        for i_epoch in range(self.max_iter):
            # ---- train ----
            self.model.train()
            loss_train = 0.0

            for X_b, y_b in ldr_train:
                X_b = X_b.to(self.device, dtype=torch.float32)
                y_b = y_b.to(self.device, dtype=torch.float32)
                y_b = self._ensure_y_3d(y_b)  # (B,T,C)

                y_flat = self.model(X_b)  # (B, T*C)
                y_pred = y_flat.view(-1, self.T, self.C)  # (B,T,C)

                mse = torch.mean((y_pred - y_b) ** 2)

                optimizer.zero_grad(set_to_none=True)
                mse.backward()

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                optimizer.step()
                loss_train += mse.item()

            loss_train /= max(len(ldr_train), 1)

            # ---- valid ----
            self.model.eval()
            loss_valid = 0.0
            with torch.no_grad():
                for X_b, y_b in ldr_valid:
                    X_b = X_b.to(self.device, dtype=torch.float32)
                    y_b = y_b.to(self.device, dtype=torch.float32)
                    y_b = self._ensure_y_3d(y_b)

                    y_flat = self.model(X_b)
                    y_pred = y_flat.view(-1, self.T, self.C)

                    mse = torch.mean((y_pred - y_b) ** 2)
                    loss_valid += mse.item()

            loss_valid /= max(len(ldr_valid), 1)

            if print_info:
                print(f"Epoch {i_epoch + 1}: train MSE={loss_train:.6f}, valid MSE={loss_valid:.6f}")

            # ---- early stopping ----
            if loss_valid < best_loss:
                best_loss = loss_valid
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                n_epochs_no_improve = 0
            else:
                n_epochs_no_improve += 1
                if n_epochs_no_improve >= self.patience:
                    if print_info:
                        print("Early stopping triggered")
                    break

        self.model.load_state_dict(best_state_dict)
        return self

    def predict(self, ds):
        # Uses ds[:] convention like your RidgeEncoder
        X, y = ds[:]
        X_t = torch.from_numpy(X).to(self.device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            y_flat = self.model(X_t).detach().cpu().numpy()  # (N, T*C)

        y_pred = y_flat.reshape(-1, self.T, self.C)  # (N,T,C)

        # match original dimensionality
        if np.asarray(y).ndim == 2 and self.C == 1:
            return y_pred[..., 0]  # (N,T)
        return y_pred  # (N,T,C)

    def score_r2(self, ds):
        X_true, y_true = ds[:]
        y_pred = self.predict(ds)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    def score_rmse(self, ds):
        X_true, y_true = ds[:]
        y_pred = self.predict(ds)
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
