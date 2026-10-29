"""Physics-informed training of a two-input circuit u(x, t) for (1+1)D PDEs.

The same idea as solver.py, one dimension up.  Each coordinate is rescaled to
[-pi/2, pi/2] and re-uploaded in every layer as RY(x~) and RX(t~) on every wire,
followed by the MS-style body (local Rot + nearest-neighbour XX).  Initial and
boundary data enter as soft penalties, exactly as in the ODE solver; there are
no hard constraints.

Reference: exact Fourier-sine series of the Gaussian initial pulse.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

jax.config.update("jax_enable_x64", True)


# ── problems (exactly as specified in the cited papers) ──────────────────────

def heat_reference(x, t, alpha=0.3, sigma2=0.2, modes=80):
    """T_t = alpha T_xx on [-pi/2, pi/2], T = 0 at both ends,
    T(x, 0) = exp(-(x + pi/8)^2 / (2 sigma2)) / 2  (Panichi et al. 2026)."""
    xq = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    t0 = 0.5 * np.exp(-(xq + np.pi / 8) ** 2 / (2 * sigma2))
    k = np.arange(1, modes + 1)
    phi = np.sin(k[:, None] * (xq[None, :] + np.pi / 2))
    b = (2 / np.pi) * np.trapezoid(t0[None, :] * phi, xq, axis=1)
    X, Tt = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    out = np.zeros(X.shape)
    for kk, bk in zip(k, b):
        out += bk * np.exp(-alpha * kk ** 2 * Tt) * np.sin(kk * (X + np.pi / 2))
    return out


def _initial_boundary(initial):
    """u(x, t0) = initial(x); u = 0 at both x ends."""
    def conditions(x0, x1, t0, t1, n=40, nt=12):
        xi = np.linspace(x0, x1, n)
        tb = np.linspace(t0, t1, nt + 1)[1:]
        return [(xi, np.full_like(xi, t0), lambda d, x, t: d["u"] - initial(x)),
                (np.r_[np.full_like(tb, x0), np.full_like(tb, x1)], np.r_[tb, tb],
                 lambda d, x, t: d["u"])]
    return conditions


PDES = {
    "heat": dict(
        label="Panichi et al. heat equation",
        x=(-np.pi / 2, np.pi / 2), t=(0.0, 0.5),
        residual=lambda d, x, t: d["ut"] - 0.3 * d["uxx"],
        conditions=_initial_boundary(lambda x: 0.5 * jnp.exp(-(x + np.pi / 8) ** 2 / 0.4)),
        reference=heat_reference,
        band=(-0.05, 0.55),
    ),
}


# ── circuit ──────────────────────────────────────────────────────────────────

def make_model(n_qubits=4, n_layers=3):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def circuit(xs, ts, w):
        for layer in w:
            for q in range(n_qubits):
                qml.RY(xs, wires=q)
                qml.RX(ts, wires=q)
            for q in range(n_qubits):
                qml.Rot(*layer[3 * q:3 * q + 3], wires=q)
            for e in range(n_qubits - 1):
                qml.IsingXX(layer[3 * n_qubits + e], wires=[e, e + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def solve_pde(name, n_qubits=4, n_layers=3, iterations=2000, stepsize=0.01,
              nx=24, nt=12, seed=0, verbose=False,
              condition_weight=1.0):
    p = PDES[name]
    (x0, x1), (t0, t1) = p["x"], p["t"]
    lo, hi = p["band"]
    circuit = make_model(n_qubits, n_layers)

    def u(x, t, w):
        xs = np.pi * (x - x0) / (x1 - x0) - np.pi / 2
        ts = np.pi * (t - t0) / (t1 - t0) - np.pi / 2
        return lo + (hi - lo) * (1 + circuit(xs, ts, w)) / 2

    ux, ut = jax.grad(u, 0), jax.grad(u, 1)
    uxx, utt = jax.grad(ux, 0), jax.grad(ut, 1)

    def derivs(x, t, w):        # unused entries are removed by XLA
        return dict(u=u(x, t, w), ux=ux(x, t, w), ut=ut(x, t, w),
                    uxx=uxx(x, t, w), utt=utt(x, t, w))

    gx, gt = np.meshgrid(np.linspace(x0, x1, nx + 2)[1:-1], np.linspace(t0, t1, nt + 1)[1:])
    xc, tc = jnp.asarray(gx.ravel()), jnp.asarray(gt.ravel())
    conds = [(jnp.asarray(cx), jnp.asarray(ct), fn) for cx, ct, fn in p["conditions"](x0, x1, t0, t1)]

    def loss(w):
        r = jax.vmap(lambda x, t: p["residual"](derivs(x, t, w), x, t))(xc, tc)
        out = jnp.mean(r ** 2)
        for cx, ct, fn in conds:
            out += condition_weight * jnp.mean(jax.vmap(lambda x, t: fn(derivs(x, t, w), x, t))(cx, ct) ** 2)
        return out

    step = jax.jit(jax.value_and_grad(loss))
    w = jnp.asarray(np.random.default_rng(seed).uniform(-0.1, 0.1, (n_layers, 4 * n_qubits - 1)))
    m = v = jnp.zeros_like(w)
    hist = []
    for it in range(1, iterations + 1):
        val, g = step(w)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - stepsize * (m / (1 - 0.9 ** it)) / (jnp.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        hist.append(float(val))
        if verbose and it % 200 == 0:
            print(f"  {name} it {it}  loss {float(val):.3e}", flush=True)

    ex, et = np.meshgrid(np.linspace(x0, x1, 81), np.linspace(t0, t1, 41))
    pred = np.asarray(jax.jit(jax.vmap(lambda x, t: u(x, t, w)))(jnp.asarray(ex.ravel()),
                                                                 jnp.asarray(et.ravel()))).reshape(ex.shape)
    ref = p["reference"](ex, et)
    return dict(x=ex, t=et, u=pred, ref=ref, loss=np.array(hist), weights=np.asarray(w),
                rel_l2=float(np.linalg.norm(pred - ref) / np.linalg.norm(ref)))

