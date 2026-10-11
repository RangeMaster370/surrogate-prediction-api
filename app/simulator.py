"""Synthetic stand-in for an expensive simulation: transformer hot-spot temperature.

Inputs (x): ambient temperature [degC], number of EVs charging, battery-shifted share [0..1].
The 'simulator' is a small IEC 60076-7 style steady-state hot-spot calculation with a
smooth load profile, plus Gaussian measurement noise. It is a teaching stand-in, not
data from any employer or client.
"""
import numpy as np

BOUNDS = np.array([[-10.0, 35.0],    # ambient degC
                   [0.0, 250.0],     # EVs charging
                   [0.0, 1.0]])      # share of charging shifted by battery
NAMES = ["ambient_degC", "n_evs", "shifted_share"]
NOISE_SD = 1.0

def true_hotspot(x):
    x = np.atleast_2d(x)
    amb, n, s = x[:, 0], x[:, 1], x[:, 2]
    base_kva, rated = 330.0, 630.0
    ev_kva = 3.2 * n * (1.0 - 0.7 * s)               # 3.2 kVA per EV at the peak, shifted share relieves it
    k = np.sqrt((base_kva ** 2 + ev_kva ** 2 + 0.5 * base_kva * ev_kva)) / rated
    d_to, d_hs, R, x_exp, y_exp = 60.0, 30.0, 5.0, 0.8, 1.6
    top_oil = amb + d_to * ((1 + R * k ** 2) / (1 + R)) ** x_exp
    return top_oil + d_hs * k ** y_exp

def prior_mean(x):
    """Deliberately simplified physics: straight load-squared heating, no oil-exponent effects."""
    x = np.atleast_2d(x)
    amb, n, s = x[:, 0], x[:, 1], x[:, 2]
    k = (330.0 + 3.2 * n * (1.0 - 0.7 * s)) / 630.0
    return amb + 62.0 * k ** 1.8

def simulate(x, rng):
    x = np.atleast_2d(x)
    return true_hotspot(x) + rng.normal(0.0, NOISE_SD, size=len(x))

def to_unit(x):
    return (np.atleast_2d(x) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])

def from_unit(u):
    return BOUNDS[:, 0] + np.atleast_2d(u) * (BOUNDS[:, 1] - BOUNDS[:, 0])

def sample_unit(n, rng):
    """Latin hypercube in the unit cube."""
    d = len(BOUNDS)
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u
