"""Self-checks; run with `python tests/test_core.py` (no pytest needed)."""

import numpy as np

from qoqpinn.pde import heat_reference
from qoqpinn.problems import LITERATURE, PROBLEMS
from qoqpinn.solver import solve


def test_exact_solutions_solve_their_odes():
    for name, p in {**PROBLEMS, **LITERATURE}.items():
        t = np.linspace(*p["region"], 401)[5:-5]
        h = 1e-4
        u = p["exact"](t)
        up = (p["exact"](t + h) - p["exact"](t - h)) / (2 * h)
        upp = (p["exact"](t + h) - 2 * u + p["exact"](t - h)) / h ** 2
        r = np.asarray(p["residual"](t, u, up, upp))
        assert np.max(np.abs(r) / (1 + np.abs(up))) < 1e-4, name
        for t0, v in p["dirichlet"]:
            assert abs(p["exact"](t0) - v) < 1e-12, name


def test_heat_reference():
    x = np.linspace(-np.pi / 2, np.pi / 2, 201)
    u0 = heat_reference(x, 0.0)
    assert np.max(np.abs(u0 - 0.5 * np.exp(-(x + np.pi / 8) ** 2 / 0.4))) < 2e-2   # pulse, up to edge mismatch
    assert abs(heat_reference(np.pi / 2, 0.3)) < 1e-12                               # boundary
    h, t = 1e-3, 0.2
    ut = (heat_reference(x, t + h) - heat_reference(x, t - h)) / (2 * h)
    dx = x[1] - x[0]
    uxx = np.gradient(np.gradient(heat_reference(x, t), dx), dx)
    assert np.max(np.abs(ut - 0.3 * uxx)[5:-5]) < 1e-3                               # PDE


def test_training_reduces_error():
    r = solve("rabi", arch="ms", n_qubits=2, n_layers=2, iterations=300,
              stepsize=0.05, n_points=40)
    assert r["loss"][-1] < 0.1 * r["loss"][0]
    assert r["rel_l2"] < 0.1, r["rel_l2"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
