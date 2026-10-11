"""Gaussian-process regression in NumPy: ARD RBF kernel, optional physics prior mean."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

def rbf(A, B, ls, var):
    d = (A[:, None, :] - B[None, :, :]) / ls
    return var * np.exp(-0.5 * np.sum(d * d, axis=2))

class GP:
    def __init__(self, prior_mean=None):
        self.prior_mean = prior_mean          # callable on unit-cube inputs, or None
        self.ls = None; self.var = None; self.noise = None

    def _m(self, X):
        return np.zeros(len(X)) if self.prior_mean is None else self.prior_mean(X)

    def _nll(self, theta, X, r):
        ls = np.exp(theta[:-2]); var = np.exp(theta[-2]); noise = np.exp(theta[-1])
        K = rbf(X, X, ls, var) + (noise + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, r)
        return 0.5 * r @ a + np.sum(np.log(np.diag(c[0]))) + 0.5 * len(X) * np.log(2 * np.pi)

    def fit(self, X, y, restarts=3, seed=0):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.X, self.y = X, y
        r = y - self._m(X)
        d = X.shape[1]
        rng = np.random.default_rng(seed)
        sd = max(np.std(r), 1e-3)
        best = None
        lo = np.r_[np.full(d, np.log(0.05)), np.log(1e-4 * sd**2), np.log(1e-5 * sd**2)]
        hi = np.r_[np.full(d, np.log(20.0)), np.log(1e3 * sd**2 + 1.0), np.log(10 * sd**2 + 1.0)]
        for i in range(restarts):
            t0 = np.r_[np.log(rng.uniform(0.3, 2.0, d)), np.log(sd**2), np.log(0.05 * sd**2 + 1e-3)]
            res = minimize(self._nll, t0, args=(X, r), method="L-BFGS-B", bounds=list(zip(lo, hi)))
            if best is None or res.fun < best.fun:
                best = res
        t = best.x
        self.ls, self.var, self.noise = np.exp(t[:d]), np.exp(t[d]), np.exp(t[d + 1])
        K = rbf(X, X, self.ls, self.var) + (self.noise + 1e-8) * np.eye(len(X))
        self._c = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._c, r)
        return self

    def set_data(self, X, y):
        self.X, self.y = X, y
        K = rbf(X, X, self.ls, self.var) + (self.noise + 1e-8) * np.eye(len(X))
        self._c = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._c, y - self._m(X))

    def predict(self, Xs):
        """Return predictive mean and standard deviation of the latent function."""
        Xs = np.atleast_2d(Xs)
        Ks = rbf(Xs, self.X, self.ls, self.var)
        mu = self._m(Xs) + Ks @ self._alpha
        v = cho_solve(self._c, Ks.T)
        var = np.maximum(self.var - np.sum(Ks * v.T, axis=1), 1e-12)
        return mu, np.sqrt(var)

    def predict_obs_std(self, Xs):
        _, s = self.predict(Xs)
        return np.sqrt(s**2 + self.noise)
