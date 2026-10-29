"""Variational qubit circuits u(t) = <Z_0>.

  ms        one encoding layer, then L x [local Rot, nearest-neighbour XX]
  hea       one encoding layer, CNOT hardware-efficient body
  reupload  L x [RY(+-(l+1) t) re-upload, RX RY, CNOT, RZ RX, CNOT]
  trotter   L x [RY(+-(l+1) t) re-upload, Trotterized exp(-i H_l tau), CNOT chain],
            H_l = sum ZZ + XX + Y + Z with trainable couplings

The coordinate enters through RY(c_g t); each circuit is written against the
vector of encoding angles c_g t.
"""

import numpy as np
import pennylane as qml

ARCHITECTURES = ("ms", "hea", "reupload", "trotter")  # qubit circuits
TROTTER_TIME, TROTTER_STEPS = 0.5, 2


def encoding_coeffs(arch, n_qubits, n_layers=3):
    """d(angle_g)/dt for every encoding gate g, in circuit order."""
    n, L = n_qubits, n_layers
    if arch == "hea":
        return np.ones(n)
    if arch == "ms":
        return np.array([(-1.0) ** w for w in range(n)])
    if arch == "reupload":
        return np.array([(-1.0) ** l * (l + 1) for l in range(L) for _ in range(n)])
    return np.array([(-1.0) ** w * (l + 1) for l in range(L) for w in range(n)])


def _chain(n):
    for i in range(n - 1):
        qml.CNOT(wires=[i, i + 1])


def _hea(angles, weights, n):
    for i in range(n):
        qml.RY(angles[i], wires=i)
    for i in range(n):
        qml.Hadamard(wires=i)
        qml.RX(weights[i], wires=i)
        qml.RY(weights[i + n], wires=i)
    _chain(n)
    for i in range(n):
        qml.Hadamard(wires=i)
        qml.RZ(weights[i + 2 * n], wires=i)
        qml.RX(weights[i + 3 * n], wires=i)
    _chain(n)
    for i in range(n):
        qml.Hadamard(wires=i)


def _ms(angles, weights, n):
    """exp(-i theta/2 X x X) is the Molmer-Sorensen gate, so every two-qubit
    gate here is one native trapped-ion operation."""
    for w in range(n):
        qml.RY(angles[w], wires=w)
    for layer in weights:
        for w in range(n):
            qml.Rot(*layer[3 * w:3 * w + 3], wires=w)
        for e in range(n - 1):
            qml.IsingXX(layer[3 * n + e], wires=[e, e + 1])


def _reupload(angles, weights, n):
    """Data re-uploading (Perez-Salinas et al., Quantum 4, 226)."""
    for l, layer in enumerate(weights):
        for i in range(n):
            qml.RY(angles[l * n + i], wires=i)
        for i in range(n):
            qml.RX(layer[2 * i], wires=i)
            qml.RY(layer[2 * i + 1], wires=i)
        _chain(n)
        for i in range(n):
            qml.RZ(layer[2 * n + 2 * i], wires=i)
            qml.RX(layer[2 * n + 2 * i + 1], wires=i)
        _chain(n)


def _trotter(angles, weights, n):
    """Re-uploaded encoding + first-order Trotterized evolution under a trainable
    Ising-type Hamiltonian, the gate sequence a digital ion simulator runs:

        H_l = sum_i (J_i Z_i Z_{i+1} + K_i X_i X_{i+1}) + sum_i (b_i Y_i + h_i Z_i)
        exp(-i H_l tau) ~ [prod_terms exp(-i c P tau/s)]^s

    Each factor is one native gate, exp(-i c P dt) = IsingZZ/IsingXX/RY/RZ(2 c dt).
    Written out gate by gate (not qml.ApproxTimeEvolution) because the
    Hamiltonian object makes XLA compilation of the nested t-derivatives
    take tens of minutes.
    """
    dt = TROTTER_TIME / TROTTER_STEPS
    for l, c in enumerate(weights):
        for w in range(n):
            qml.RY(angles[l * n + w], wires=w)
        for _ in range(TROTTER_STEPS):
            for i in range(n - 1):
                qml.IsingZZ(2 * c[2 * i] * dt, wires=[i, i + 1])
                qml.IsingXX(2 * c[2 * i + 1] * dt, wires=[i, i + 1])
            for i in range(n):
                qml.RY(2 * c[2 * (n - 1) + 2 * i] * dt, wires=i)
                qml.RZ(2 * c[2 * (n - 1) + 2 * i + 1] * dt, wires=i)
        _chain(n)


_BODIES = {"ms": _ms, "hea": _hea, "reupload": _reupload, "trotter": _trotter}


def init_weights(arch, n_qubits, n_layers=3, seed=0):
    rng = np.random.default_rng(seed)
    n, L = n_qubits, n_layers
    if arch == "hea":
        return rng.uniform(0, 2 * np.pi, size=4 * n)
    if arch == "ms":
        return rng.uniform(-0.1, 0.1, size=(L, 4 * n - 1))
    if arch == "reupload":
        return rng.uniform(0, 2 * np.pi, size=(L, 4 * n))
    return 0.1 * rng.standard_normal(size=(L, 4 * n - 2))


def make_circuit(arch, n_qubits, interface="jax"):
    """Return ``f(angles, weights) -> <Z_0>`` as a QNode on an exact statevector."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {ARCHITECTURES}, got {arch!r}")
    body = _BODIES[arch]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=interface)
    def circuit(angles, weights):
        body(angles, weights, n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_model(arch, n_qubits, n_layers=3, interface="jax"):
    """``qnn(t, weights)`` with angles = coeffs * t."""
    circuit = make_circuit(arch, n_qubits, interface)
    coeffs = encoding_coeffs(arch, n_qubits, n_layers)

    def qnn(t, weights):
        return circuit(coeffs * t, weights)

    return qnn
