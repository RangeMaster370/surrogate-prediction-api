"""Train a GP surrogate on simulator runs and expose validated prediction."""
import hashlib
import numpy as np
from . import simulator as sim
from .gp import GP

class Surrogate:
    def __init__(self, n_runs=40, seed=0):
        rng = np.random.default_rng(seed)
        U = sim.sample_unit(n_runs, rng)
        y = sim.simulate(sim.from_unit(U), rng)
        self.gp = GP().fit(U, y, restarts=3, seed=seed)
        self.n_runs, self.seed = n_runs, seed
        self.version = hashlib.sha256(np.r_[U.ravel(), y].tobytes()).hexdigest()[:10]

    def predict(self, rows):
        """rows: list of [ambient_degC, n_evs, shifted_share]. Returns mean and 95 % half-width."""
        U = sim.to_unit(np.asarray(rows, float))
        mu, _ = self.gp.predict(U)
        so = self.gp.predict_obs_std(U)
        return mu, 1.96 * so
