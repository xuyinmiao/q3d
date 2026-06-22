import os
from datetime import datetime
import pennylane as qml
import torch
import torch.nn as nn

# ===================== 设置结果保存路径 =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

save_dir = os.path.join(project_root, "other_pictures")
os.makedirs(save_dir, exist_ok=True)

# ===================== 量子线路（可替换） =====================
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs, conv1_weights, conv2_weights, dense_weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.Rot(*conv1_weights[i], wires=i)
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.Rot(*conv2_weights[i], wires=i)
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[0, n_qubits - 1])
    for i in range(n_qubits):
        qml.Rot(*dense_weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# ===================== 设置输入数据与权重参数 =====================
inputs = torch.tensor(
    [3.0831, 3.1445, 2.9018, 3.1910, 3.0092, 3.0426, 3.1109, 3.4162],
    dtype=torch.float32
)
conv1_weights = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
conv2_weights = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
dense_weights = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)

# ===================== 保存图像（png/svg/pdf） =====================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
png_filename = os.path.join(save_dir, f"quantum_circuit_{timestamp}.png")
svg_filename = os.path.join(save_dir, f"quantum_circuit_{timestamp}.svg")  
pdf_filename = os.path.join(save_dir, f"quantum_circuit_{timestamp}.pdf")  

fig, ax = qml.draw_mpl(circuit, style="pennylane", fontsize="xx-large")(
    inputs, conv1_weights, conv2_weights, dense_weights
)

# fig.savefig(png_filename, bbox_inches="tight", dpi=600)
# fig.savefig(pdf_filename, bbox_inches="tight")
fig.savefig(svg_filename, bbox_inches="tight")

print("量子线路图已保存为：")
# print(png_filename)
# print(pdf_filename)
print(svg_filename)
