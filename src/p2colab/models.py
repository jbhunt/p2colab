import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from sklearn.linear_model import Ridge
from torch import nn
import copy

class RNNModule(nn.Module):
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

class RNNDecoder():
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
        self.model = RNNModule(
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
    
    def reset(self):
        self._return_to_initial_state()
    
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
        y_pred = self.predict(ds).flatten()
        floor = np.sum(np.power(y_true - y_true.mean(), 2))
        residual = np.sum(np.power(y_true - y_pred, 2))
        r2 = 1 - (residual / floor)

        return r2
    
class LinearDecoder():
    """
    """

    def __init__(self, alpha=1.0, max_iter=1000):
        """
        """

        self.alpha = alpha
        self.model = Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=True)

        return
    
    def _return_to_initial_state(self):
        """
        """

        self.model = Ridge(alpha=self.alpha)

        return
    
    def reset(self):
        self._return_to_initial_state()

    def fit(self, ds_train, ds_valid=None, print_info=None):
        """
        """

        X, y = ds_train[:]
        N, T, C = X.shape
        X = X.reshape(N, T * C)
        self.model.fit(X, y)

        return

    def predict(self, ds):
        """
        """

        X, y = ds[:]
        N, T, C = X.shape
        X = X.reshape(N, T * C)
        out = self.model.predict(X)

        return out
    
    def score(self, ds):
        """
        """

        _, y_true = ds[:]
        y_pred = self.predict(ds)
        rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2))).item()

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
    
class MLP(nn.Module):
    """
    """

    def __init__(self, input_size, hidden_layer_sizes=[128, 16], dropout=0.1):
        """
        """

        super().__init__()
        modules = list()
        fc = nn.Linear(in_features=input_size, out_features=hidden_layer_sizes[0])
        modules.append(fc)
        relu = nn.ReLU()
        modules.append(relu)
        modules.append(nn.Dropout(dropout))
        for i, n1 in enumerate(hidden_layer_sizes[:-1]):
            n2 = hidden_layer_sizes[i + 1]
            fc = nn.Linear(in_features=n1, out_features=n2)
            modules.append(fc)
            relu = nn.ReLU()
            modules.append(relu)
            modules.append(nn.Dropout(dropout))
        fc = nn.Linear(in_features=hidden_layer_sizes[-1], out_features=1)
        modules.append(fc)
        self.seq = nn.Sequential(*modules)

        return
    
    def forward(self, X):
        """
        """

        out = self.seq(X)

        return out
    
class MLPDecoder():
    """
    """

    def __init__(self, hidden_layer_sizes=[128, 64], dropout=0.1, max_iter=1000, lr=0.0001, batch_size=16, patience=10, device=None):
        """
        """

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.max_iter = max_iter
        self.lr = lr
        self.model = None
        self.batch_size = batch_size
        self.patience = patience
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        return
    
    def reset(self):
        """
        """

        # TODO: Implement initial state reset (if it becomes important)

        return
    
    def fit(self, ds_train, ds_valid=None, print_info=True):
        """
        """

        dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        if ds_valid is not None:
            dl_valid = DataLoader(ds_valid, batch_size=self.batch_size)
        else:
            dl_valid = None
        N, T, C = ds_train.X.shape
        D = T * C
        self.model = MLP(input_size=D, hidden_layer_sizes=self.hidden_layer_sizes, dropout=self.dropout).to(device=self.device)

        #
        loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        best_loss = np.inf
        counter = 0
        best_state_dict = None
        for i_epoch in range(self.max_iter):

            #
            self.model.train()
            batch_loss_train = list()
            for X_true, y_true in dl_train:
                B = X_true.size()[0]
                X_true = torch.as_tensor(X_true, dtype=torch.float32, device=self.device)
                X_true = torch.reshape(X_true, (B, D))
                y_true = torch.as_tensor(y_true, dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_true).flatten()
                loss = loss_fn(y_true, y_pred)
                batch_loss_train.append(loss.item())
                loss.backward()
                optimizer.step()
            batch_loss_train = np.mean(batch_loss_train)
            
            #
            if ds_valid is not None:
                self.model.eval()
                batch_loss_valid = list()
                with torch.no_grad():
                    for X_true, y_true in dl_valid:
                        B = X_true.size()[0]
                        X_true = torch.as_tensor(X_true, dtype=torch.float32, device=self.device)
                        X_true = torch.reshape(X_true, (B, D))
                        y_true = torch.as_tensor(y_true, dtype=torch.float32, device=self.device)
                        y_pred = self.model(X_true).flatten()
                        loss = loss_fn(y_true, y_pred)
                        batch_loss_valid.append(loss.item())
                batch_loss_valid = np.mean(batch_loss_valid)
                if print_info:
                    end = "\r" if (i_epoch + 1) < self.max_iter else "\n"
                    print(f"Epoch {i_epoch} out of {self.max_iter}: Loss = {batch_loss_valid:.6f}", end=end)

            if ds_valid is not None:
                batch_loss = batch_loss_valid
            else:
                batch_loss = batch_loss_train
            if batch_loss < best_loss:
                counter = 0
                best_loss = batch_loss
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1
            if counter >= self.patience:
                if print_info:
                    print(f"Epoch {i_epoch} out of {self.max_iter}: Loss = {batch_loss:.6f}")
                    print("Early stopping condition met")
                break

        #
        self.model.load_state_dict(best_state_dict)

        return
    
    def predict(self, ds):
        """
        Notes
        -----
        X - (N, T, C)
        """

        self.model.eval()
        with torch.no_grad():
            X, y = ds[:]
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            N, T, C = X.shape
            D = T * C
            X = torch.reshape(X, (N, D))
            out = self.model(X).detach().cpu()

        return out
    
    def score_r2(self, ds):
        """
        """

        _, y_true = ds[:]
        y_pred = self.predict(ds).numpy().flatten()
        rss = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (rss / tss)

        return r2

class LinearEncoder():
    """
    Simple linear encoder with L2 penalty
    """

    def __init__(self, alpha=1.0):
        """
        """

        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

        return
    
    def _return_to_initial_state(self): return

    def reset(self):
        self._return_to_initial_state()
    
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
