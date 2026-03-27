from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class NeuralActivityProcessor():
    """
    """

    def __init__(self, n_components=None):
        """
        """

        self.n_components = n_components
        self._fit = False

        return
    
    def fit(self, X):
        """
        """

        N, T, C = X.shape
        X_new = X.reshape(N * T, C)
        self.tf1 = StandardScaler().fit(X_new)
        if self.n_components is None:
            self.tf2 = None
        else:
            self.tf2 = PCA(n_components=self.n_components)
            self.tf2.fit(self.tf1.transform(X_new))
        self._fit = True

        return
    
    def fit_transform(self, X):
        """
        """

        self.fit(X)
        out = self.transform(X)

        return out
    
    def transform(self, X):
        """
        """

        if self._fit == False:
            raise Exception("Model must be fit before applying transform")
        N, T, C = X.shape
        X_new = X.reshape(N * T, C)
        out = self.tf1.transform(X_new)
        if self.tf2 is not None:
            out = self.tf2.transform(out)
            C_out = self.n_components
        else:
            C_out = C
        out = out.reshape(N, T, C_out)

        return out