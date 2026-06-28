"""Generate an expanded quantum circuit figure (Figure 2) for the paper.

Shows the gate-level decomposition of the variational quantum layer used in
HybridQWideResNet: AngleEmbedding(Y) + StronglyEntanglingLayers(6) + PauliZ.
We expand the layers so individual gates are visible rather than collapsed.
"""
import os
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(save_dir, exist_ok=True)

n_qubits = 8
n_layers = 6
dev = qml.device("default.qubit", wires=n_qubits)


def expanded_circuit(inputs, weights):
    # Stage 1: AngleEmbedding (RY rotations)
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Stage 2: StronglyEntanglingLayers, expanded layer by layer
    for layer in range(n_layers):
        w = weights[layer]
        for q in range(n_qubits):
            qml.Rot(w[q, 0], w[q, 1], w[q, 2], wires=q)
        # StronglyEntanglingLayers entanglement pattern
        for q in range(n_qubits):
            partner = (q + 1) % n_qubits
            qml.CNOT(wires=[q, partner])
        for q in range(n_qubits):
            partner = (q + 3) % n_qubits
            qml.CNOT(wires=[q, partner])

    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]


circuit = qml.QNode(expanded_circuit, dev)

inputs = np.linspace(0.0, 2 * np.pi, n_qubits)
weights = np.random.randn(n_layers, n_qubits, 3) * 0.1

fig, ax = qml.draw_mpl(circuit, style="pennylane", fontsize=10)(inputs, weights)

out_png = os.path.join(save_dir, "quantum_circuit.png")
out_svg = os.path.join(save_dir, "quantum_circuit.svg")
fig.savefig(out_png, bbox_inches="tight", dpi=300)
fig.savefig(out_svg, bbox_inches="tight")
print("Saved:")
print(" ", out_png)
print(" ", out_svg)

print("\nASCII circuit (first part):\n")
txt = qml.draw(circuit)(inputs, weights)
print(txt[:2000])
