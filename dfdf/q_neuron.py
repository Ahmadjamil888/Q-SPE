# q_neuron.py
# Quantum–Neural Hybrid "neuron" demo
# Run: python q_neuron.py
# Requires: numpy only

import numpy as np

# --- Linear algebra helpers ---
def kron(*mats):
    out = np.array([[1.0+0j]])
    for m in mats:
        out = np.kron(out, m)
    return out

# --- Basis states ---
ZERO = np.array([[1.0+0j],[0.0+0j]])
ONE  = np.array([[0.0+0j],[1.0+0j]])

# --- Single-qubit rotations ---
def RX(theta):
    c = np.cos(theta/2.0)
    s = -1j*np.sin(theta/2.0)
    return np.array([[c, s],
                     [s, c]], dtype=complex)

def RY(theta):
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    return np.array([[c, -s],
                     [s,  c]], dtype=complex)

def RZ(theta):
    return np.array([[np.exp(-1j*theta/2.0), 0],
                     [0, np.exp(1j*theta/2.0)]], dtype=complex)

# --- Two-qubit CNOT (control q0, target q1) ---
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]], dtype=complex)

# --- Measurement projector on first qubit ---
P1_q0 = kron(np.array([[0,0],[0,1]], dtype=complex), np.eye(2, dtype=complex))

def normalize(psi):
    n = np.linalg.norm(psi)
    return psi if n == 0 else psi / n

# --- Quantum "neuron" forward pass ---
def forward(weights, x, use_entanglement=True):
    psi = kron(ZERO, ZERO)
    # Encode inputs as rotations
    U0 = RY(weights['w0']*x[0] + weights['b0'])
    U1 = RY(weights['w1']*x[1] + weights['b1'])
    psi = kron(U0, U1) @ psi
    psi = normalize(psi)

    if use_entanglement:
        E = kron(RZ(weights['e']), np.eye(2, dtype=complex))
        psi = E @ psi
        psi = CNOT @ psi
        psi = E @ psi
        psi = normalize(psi)

    U_read = kron(RY(weights['w2']), RY(weights['w3']))
    psi = U_read @ psi
    psi = normalize(psi)

    prob1 = np.real(np.conjugate(psi).T @ (P1_q0 @ psi))[0,0]
    eps = 1e-9
    return float(np.clip(prob1, eps, 1.0 - eps))

# --- Loss ---
def loss(weights, X, y, use_entanglement=True):
    preds = [forward(weights, xi, use_entanglement) for xi in X]
    preds = np.array(preds)
    y = np.array(y)
    ce = -(y*np.log(preds) + (1-y)*np.log(1-preds))
    return float(np.mean(ce)), preds

# --- Parameter-shift gradient ---
def param_shift_grad(weights, X, y, key, shift=np.pi/2, use_entanglement=True):
    w_p = dict(weights); w_m = dict(weights)
    w_p[key] = weights[key] + shift
    w_m[key] = weights[key] - shift
    Lp,_ = loss(w_p, X, y, use_entanglement)
    Lm,_ = loss(w_m, X, y, use_entanglement)
    return (Lp - Lm)/2.0

def train_qneuron(X, y, steps=600, lr=0.15, use_entanglement=True, seed=7):
    rng = np.random.default_rng(seed)
    weights = {
        'w0': rng.normal(scale=0.5),
        'w1': rng.normal(scale=0.5),
        'w2': rng.normal(scale=0.5),
        'w3': rng.normal(scale=0.5),
        'b0': rng.normal(scale=0.5),
        'b1': rng.normal(scale=0.5),
        'e' : rng.normal(scale=0.5)
    }
    keys = list(weights.keys())
    for t in range(steps):
        L,_ = loss(weights, X, y, use_entanglement)
        grads = {k: param_shift_grad(weights, X, y, k, use_entanglement=use_entanglement) for k in keys}
        for k in keys:
            weights[k] -= lr * grads[k]
        if (t+1) % (steps//10) == 0:
            print(f"step {t+1:4d}/{steps} | loss={L:.4f}")
    return weights

# --- Dataset: XOR ---
X = np.array([[0.0,0.0],
              [0.0,1.0],
              [1.0,0.0],
              [1.0,1.0]], dtype=float)
y = np.array([0,1,1,0], dtype=float)

print("\nTraining with entanglement...")
weights_ent = train_qneuron(X, y, steps=600, lr=0.15, use_entanglement=True, seed=42)
final_loss, preds = loss(weights_ent, X, y, use_entanglement=True)
print("\nFinal loss (entangled):", final_loss)
for xi, pi, yi in zip(X, preds, y):
    print(f"x={xi} -> p1={pi:.3f} (label={int(yi)})")

print("\nTraining without entanglement...")
weights_noe = train_qneuron(X, y, steps=600, lr=0.15, use_entanglement=False, seed=123)
final_loss_noe, preds_noe = loss(weights_noe, X, y, use_entanglement=False)
print("\nFinal loss (no entanglement):", final_loss_noe)
for xi, pi, yi in zip(X, preds_noe, y):
    print(f"x={xi} -> p1={pi:.3f} (label={int(yi)})")
