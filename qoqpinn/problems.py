"""Quantum-optics ODEs with closed-form solutions.

Each problem gives the residual r(t, u, u', u'') that vanishes on the solution,
initial/boundary data, the time window, the exact solution, and the output
band [lo, hi] the read-out is mapped onto: u = lo + (hi - lo)(1 + <Z_0>)/2.
Every band leaves ~10% headroom above and below the exact solution, so no
target is reachable only by a saturated read-out.
"""

import jax.numpy as jnp
import numpy as np

def _band(lo, hi, pad=0.1):
    d = pad * (hi - lo)
    return (lo - d, hi + d)


def _duffing(t, A=1.0, w=1.0, beta=1.0):
    """x'' + w^2 x + beta x^3 = 0, x(0) = A, x'(0) = 0:  x = A cn(W t | m),
    W^2 = w^2 + beta A^2, m = beta A^2 / (2 W^2)."""
    from scipy.special import ellipj
    W = np.sqrt(w ** 2 + beta * A ** 2)
    return A * ellipj(W * np.asarray(t, float), beta * A ** 2 / (2 * W ** 2))[1]


_G = 0.2                              # damped Rabi: decay rate, Omega = 1
_PSS = 0.5 / (1 + _G ** 2)            # steady-state excited population
_ALPHA, _BETA = 0.5, 1.0              # two-photon absorption: linear, 2-photon coefficients


# One ODE per mathematical family, each a quantum-optics system.
PROBLEMS = {
    "rabi": dict(
        family="linear, constant coefficients, undamped",
        label="Rabi oscillation",
        # Bloch equations at resonance: P'' = Omega^2 (1/2 - P), P = sin^2(Omega t/2).
        residual=lambda t, u, up, upp: upp + u - 0.5,
        region=(0.0, 2 * np.pi),
        dirichlet=[(0.0, 0.0)],
        neumann=[(0.0, 0.0)],
        exact=lambda t: np.sin(t / 2) ** 2,
        band=_band(0.0, 1.0),
    ),
    "damped_rabi": dict(
        family="linear, constant coefficients, damped, inhomogeneous",
        label="damped Rabi oscillation",
        # Optical Bloch equations with equal decay rates gamma for coherence and
        # population: P'' + 2 gamma P' + (Omega^2 + gamma^2) P = Omega^2 / 2.
        residual=lambda t, u, up, upp: upp + 2 * _G * up + (1 + _G ** 2) * u - 0.5,
        region=(0.0, 5.0),
        dirichlet=[(0.0, 0.0)],
        neumann=[(0.0, 0.0)],
        exact=lambda t: _PSS * (1 - np.exp(-_G * t) * (np.cos(t) + _G * np.sin(t))),
        band=_band(0.0, 0.75),
    ),
    "photon_decay": dict(
        family="linear, first order",
        label="cavity photon decay",
        # n' = -kappa n, kappa = 1.
        residual=lambda t, u, up, upp: up + u,
        region=(0.0, 4.0),
        dirichlet=[(0.0, 1.0)],
        neumann=[],
        exact=lambda t: np.exp(-t),
        band=_band(0.0, 1.0),
    ),
    "two_photon_absorption": dict(
        family="nonlinear, first order (Bernoulli)",
        label="two-photon absorption",
        # Intensity along a crystal: I' = -alpha I - beta I^2.
        residual=lambda t, u, up, upp: up + _ALPHA * u + _BETA * u ** 2,
        region=(0.0, 4.0),
        dirichlet=[(0.0, 1.0)],
        neumann=[],
        exact=lambda t: _ALPHA * np.exp(-_ALPHA * t)
        / (_ALPHA + _BETA * (1 - np.exp(-_ALPHA * t))),
        band=_band(0.0, 1.0),
    ),
    "hermite_gauss": dict(
        family="linear, variable coefficients (Sturm-Liouville)",
        label="Hermite-Gauss mode, n = 1",
        # Transverse beam mode / Fock-state wavefunction:
        # psi'' + (2n + 1 - x^2) psi = 0, psi_1 = x exp(-x^2/2).
        # psi = 0 also solves it, so two interior points anchor the amplitude.
        residual=lambda t, u, up, upp: upp + (3.0 - t ** 2) * u,
        region=(-3.0, 3.0),
        dirichlet=[(x, x * np.exp(-x ** 2 / 2)) for x in (-3.0, -1.0, 1.0, 3.0)],
        neumann=[],
        exact=lambda t: t * np.exp(-t ** 2 / 2),
        band=_band(-0.61, 0.61),
    ),
    "kerr_oscillator": dict(
        family="nonlinear oscillator (Duffing)",
        label="Kerr (anharmonic) oscillator",
        # Classical field in a Kerr cavity: x'' + w^2 x + beta x^3 = 0.
        residual=lambda t, u, up, upp: upp + u + u ** 3,
        region=(0.0, 5.0),
        dirichlet=[(0.0, 1.0)],
        neumann=[(0.0, 0.0)],
        exact=_duffing,
        band=_band(-1.0, 1.0),
    ),
    "nls_soliton": dict(
        family="nonlinear boundary-value problem (homoclinic)",
        label="NLS bright soliton",
        # i A_z + A_tt / 2 + |A|^2 A = 0 with A = u(t) e^{iz/2}: u'' - u + 2u^3 = 0.
        residual=lambda t, u, up, upp: upp - u + 2.0 * u ** 3,
        region=(-3.0, 3.0),
        dirichlet=[(-3.0, 1 / np.cosh(3.0)), (0.0, 1.0), (3.0, 1 / np.cosh(3.0))],
        neumann=[],
        exact=lambda t: 1 / np.cosh(t),
        band=_band(0.0, 1.0),
    ),
}


def relative_l2(pred, exact):
    pred, exact = np.asarray(pred), np.asarray(exact)
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


# ── Benchmarks reproduced from the literature ───────────────────────
# Same equations, conditions and domains as in the cited papers.

_W_SPRING = np.sqrt(3.0) / 2.0  # u'' + u' + u = 0: roots -1/2 +- i sqrt(3)/2

LITERATURE = {
    # Kyriienko, Paine & Elfving, PRA 103, 052416 (2021), Eq. (31):
    # u' + lambda u (kappa + tan(lambda x)) = 0, u(0) = 1, lambda = 8, kappa = 0.1,
    # x in [0, 0.9]; exact u = exp(-kappa lambda x) cos(lambda x).
    "kyriienko_ode": dict(
        family="linear, first order, variable coefficient",
        label="Kyriienko et al. ODE",
        residual=lambda t, u, up, upp: up + 8.0 * u * (0.1 + jnp.tan(8.0 * t)),
        region=(0.0, 0.9),
        dirichlet=[(0.0, 1.0)],
        neumann=[],
        exact=lambda t: np.exp(-0.8 * t) * np.cos(8.0 * t),
        band=_band(-0.75, 1.0),
    ),
    # Hunout, Laizet & Iannucci, PRA 111, 062404 (2025): damped mass-spring
    # m u'' + b u' + k u = 0 with m = b = k = 1, u(0) = 1, u'(0) = 0, t in [0, 10].
    "hunout_spring": dict(
        family="linear, constant coefficients, damped",
        label="Hunout et al. mass-spring",
        residual=lambda t, u, up, upp: upp + up + u,
        region=(0.0, 10.0),
        dirichlet=[(0.0, 1.0)],
        neumann=[(0.0, 0.0)],
        exact=lambda t: np.exp(-0.5 * t) * (np.cos(_W_SPRING * t)
                                             + 0.5 / _W_SPRING * np.sin(_W_SPRING * t)),
        band=_band(-0.15, 1.0),
    ),
    # Panichi, Corli & Prati, Phys. Rev. Applied 25, 014001 (2026), Sec. IV.1:
    # u'' + sin(4x) = 0, u(0) = u(pi/2) = 0; exact u = sin(4x)/16.
    "panichi_poisson": dict(
        family="linear boundary-value problem (1D Poisson)",
        label="Panichi et al. 1D Poisson",
        residual=lambda t, u, up, upp: upp + jnp.sin(4.0 * t),
        region=(0.0, np.pi / 2),
        dirichlet=[(0.0, 0.0), (np.pi / 2, 0.0)],
        neumann=[],
        exact=lambda t: np.sin(4.0 * t) / 16.0,
        band=_band(-0.0625, 0.0625),
    ),
}

