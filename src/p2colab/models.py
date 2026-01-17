import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

class VanillaRNN(nn.Module):
    """
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_layers=2,
        dropout=0.2,
        ):
        """
        """

        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

        return
    
    def forward(self, X):
        """
        X must have the shape N x T x U
        TODO: Consider learning some kind of simple attention or averaging over time steps (instead of just using the last hidden state)
        """

        out, (h_n, c_n) = self.rnn(X) # N x T x H (number of hidden states)
        h_last = self.dropout(h_n[-1]) # N x H
        y = self.head(h_last) # N x 1

        return y
    
class SpicyRNN(nn.Module):
    """
    """

    def __init__(
        self,
        C_in,
        output_size,
        C_out=32,
        kernel_size=5,
        hidden_size=32,
        n_layers=2,
        dropout=0.1,
        ):
        """
        """

        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
            ),
            nn.ReLU()
        )

        # RNN encoder
        self.rnn = nn.LSTM(
            input_size=C_out,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        # MLP regressor
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        return
    
    def forward(self, X):
        """
        """

        #
        X_tp = X.transpose(1, 2) # N x U (units) x T
        X_conv = self.cnn(X_tp)  # N x F (new features) x T
        X_conv = X_conv.transpose(1, 2) # N x T x F
        out, (h_n, c_n) = self.rnn(X_conv) # out is N x T x H (number of hidden states)
        h_summary = out.mean(dim=1) # N x H
        h_summary = self.dropout(h_summary) # N x H
        y = self.head(h_summary) # N x output size

        return y
    
class Seq2SeqDecoder():
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
        batch_size=32,
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
        # self.model = VanillaRNN(
        #     input_size=input_size,
        #     output_size=output_size
        # ).to(self.device)
        self.model = SpicyRNN(
            C_in=input_size,
            output_size=output_size,
            C_out=C_out,
            dropout=dropout,
            kernel_size=kernel_size
        )
        # self.model = DecoderOnlyTransformer(
        #     d_in=input_size,
        #     d_out=70,
        #     d_model=64,
        #     n_heads=8,
        #     n_layers=2,
        #     dim_ff=256,
        #     dropout=dropout,
        #     max_seq_len=512,
        #     head_type="attn_pool",
        # ).to(self.device)
        self._init_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.patience = patience
        self.task_type = task_type # TODO: Extend the model to handle classification tasks

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

        ldr_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        ldr_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=False)
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
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                n_epochs_no_improve = 0
            else:
                n_epochs_no_improve += 1
            if n_epochs_no_improve >= self.patience:
                if print_info:
                    print("Early stopping triggered")
                break

        # Load the model from the best epoch
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
    
def _make_rasied_cosine_basis_matrix(
    T,
    K,
    t_min=0.0,
    t_max=1.0,
    width_scale=1.0,
    device=None,
    dtype=torch.float32
    ):
    """
    """

    device = "cpu" if device is None else device
    t = torch.linspace(t_min, t_max, T, device=device, dtype=dtype)  # (T,)
    centers = torch.linspace(t_min, t_max, K, device=device, dtype=dtype)  # (K,)
    dc = centers[1] - centers[0]
    w = width_scale * dc
    x = (t[:, None] - centers[None, :]) / w
    Phi = torch.zeros((T, K), device=device, dtype=dtype)
    mask = (x.abs() <= 1.0)
    Phi[mask] = 0.5 * (1.0 + torch.cos(torch.pi * x[mask]))
    Phi = Phi / (Phi.norm(dim=0, keepdim=True) + 1e-8)

    return Phi
    
class FancyRegressor(nn.Module):
    """
    """

    def __init__(self, input_size, T, C, K=7, width_scale=1.0):
        """
        """

        super().__init__()
        Phi = _make_rasied_cosine_basis_matrix(T, K, width_scale=width_scale)
        self.register_buffer("Phi", Phi)
        self.C = C
        self.K = K
        self.linear = nn.Linear(in_features=input_size + 1, out_features=K * C, bias=False, dtype=torch.float32)

        return
    
    def forward(self, X):
        """
        """

        N, _ = X.shape
        X_new = torch.cat([torch.ones(N, 1, device=X.device, dtype=torch.float32), X], dim=1)  # N x F + 1

        coef = self.linear(X_new)              # N x C * K
        coef = coef.view(N, self.C, self.K)    # N x C x K

        Y_hat = coef @ self.Phi.T              # N x C x T
        Y_hat = Y_hat.transpose(1, 2)          # N x T x C

        return Y_hat
    
class VanillaMLP(nn.Module):
    """
    Simple feedforward MLP: input -> hidden layers -> output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes=(128,),
        activation=nn.ReLU,
        dropout: float = 0.1,
        ):
        super().__init__()

        layers = []
        in_dim = input_size

        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_size))
        self.seq = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.seq(X)


class MLPEncoder:
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

        self.model = VanillaMLP(
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
    
class RidgeEncoder():
    """
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
        self.model = FancyRegressor(input_size=F, T=T, C=C, K=self.K, width_scale=self.width_scale)
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