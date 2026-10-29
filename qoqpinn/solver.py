"""Physics-informed training of a circuit u(t) (Kyriienko, Paine & Elfving 2021).

u' and u'' come from exact autodiff through the statevector simulation; on
hardware they would come from the parameter-shift rule.
Loss = mean squared residual on a collocation grid + boundary penalties.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .circuits import init_weights, make_model
from .problems import PROBLEMS, relative_l2

jax.config.update("jax_enable_x64", True)


def read_out(z, band):
    lo, hi = band
    return lo + (hi - lo) * (1.0 + z) / 2.0


N_EXTRA = 4  # padded slots for boundary points, so every problem shares one grid shape
_KERNELS = {}


def _kernels(arch, n_qubits, n_layers):
    """Jitted (z, z', z'') on a grid and its VJP, compiled once per circuit.

    Compiling the nested t-derivatives of a deep circuit takes minutes, so the
    circuit part takes the grid as an argument and is shared by every problem
    and seed; only the cheap residual algebra is compiled per problem.
    """
    key = (arch, n_qubits, n_layers)
    if key not in _KERNELS:
        qnn = make_model(arch, n_qubits, n_layers)
        d1 = jax.grad(qnn)
        d2 = jax.grad(d1)

        def point(x, w):
            return jnp.stack([qnn(x, w), d1(x, w), d2(x, w)])

        def zs(w, t):
            return jax.vmap(lambda x: point(x, w))(t)

        fwd = jax.jit(zs)
        bwd = jax.jit(lambda w, t, ct: jax.vjp(lambda v: zs(v, t), w)[1](ct)[0])
        _KERNELS[key] = (fwd, bwd)
    return _KERNELS[key]


def solve(problem, arch="ms", n_qubits=4, n_layers=3, iterations=2000,
          stepsize=0.01, n_points=100, seed=0, verbose=False,
          rescale=False, condition_weight=1.0):
    """Train one model.  Returns a dict with t, u, exact, rel_l2, loss, weights.

    rescale: feed the circuit x~ = pi (x - x0)/(x1 - x0) - pi/2 in [-pi/2, pi/2]
        instead of x, as the literature solvers do; derivatives are converted
        with the chain rule.  Off by default (the optics benchmarks use raw x).
    condition_weight: multiplies the value/slope penalty terms (1 = unweighted).
    """
    p = PROBLEMS[problem] if isinstance(problem, str) else problem
    fwd, bwd = _kernels(arch, n_qubits, n_layers)
    lo, hi = p["band"]
    s = (hi - lo) / 2

    nd, nn = len(p["dirichlet"]), len(p["neumann"])
    assert nd + nn <= N_EXTRA
    t_col = np.linspace(*p["region"], n_points)
    extra = [a for a, _ in p["dirichlet"]] + [a for a, _ in p["neumann"]]
    t_all = jnp.asarray(np.concatenate([t_col, extra, np.zeros(N_EXTRA - nd - nn)]))
    x0, x1 = p["region"]
    c = np.pi / (x1 - x0) if rescale else 1.0       # d x~ / d x
    t_in = c * (t_all - x0) - np.pi / 2 if rescale else t_all
    v_d = jnp.array([b for _, b in p["dirichlet"]])
    v_n = jnp.array([b for _, b in p["neumann"]])

    def loss_of_z(zs):
        u, up, upp = lo + s * (1 + zs[:, 0]), s * c * zs[:, 1], s * c ** 2 * zs[:, 2]
        k = slice(0, n_points)
        out = jnp.mean(p["residual"](t_all[k], u[k], up[k], upp[k]) ** 2)
        if nd:
            out += condition_weight * jnp.mean((u[n_points:n_points + nd] - v_d) ** 2)
        if nn:
            out += condition_weight * jnp.mean((up[n_points + nd:n_points + nd + nn] - v_n) ** 2)
        return out

    loss_grad = jax.jit(jax.value_and_grad(loss_of_z))

    def step(w):
        val, ct = loss_grad(fwd(w, t_in))
        return val, bwd(w, t_in, ct)

    w = jnp.asarray(init_weights(arch, n_qubits, n_layers, seed))
    m, v = jnp.zeros_like(w), jnp.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for it in range(1, iterations + 1):
        val, g = step(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - stepsize * (m / (1 - b1 ** it)) / (jnp.sqrt(v / (1 - b2 ** it)) + eps)
        history.append(float(val))
        if verbose and it % 200 == 0:
            print(f"  {problem} it {it:5d}  loss {float(val):.3e}")

    pred = np.asarray(read_out(fwd(w, t_in)[:n_points, 0], p["band"]))
    exact = p["exact"](t_col)
    return dict(t=t_col, u=pred, exact=exact, rel_l2=relative_l2(pred, exact),
                loss=np.array(history), weights=np.asarray(w))
