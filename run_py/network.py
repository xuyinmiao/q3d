import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_quantum_layer(n_qubits=8, n_layers=3):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)

class CNN(nn.Module):
    def __init__(self, latent_dim=8, n_classes=10):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flat_size = 32 * 7 * 7
        self.feature_reduction = nn.Linear(self.flat_size, latent_dim)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_size)
        x = torch.sigmoid(self.feature_reduction(x))
        return self.classifier(x)

class HybridQCNN(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3, n_classes=10):
        super(HybridQCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flat_size = 32 * 7 * 7
        self.feature_reduction = nn.Linear(self.flat_size, n_qubits)
        self.quantum_layer = make_quantum_layer(n_qubits, n_layers)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_qubits),
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_size)
        x = torch.sigmoid(self.feature_reduction(x))
        x = self.quantum_layer(x)
        return self.classifier(x)

############################################################################################

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.shortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if self.shortcut is not None:
            return self.shortcut(x) + out
        else:
            return x + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):  
    def __init__(self, n_classes=10, depth=28, widen_factor=10, drop_rate=0.3, latent_dim=8):
        super(WideResNet, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.feature_reduction = nn.Linear(self.nChannels, latent_dim)
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(-1, self.nChannels)
        out = torch.sigmoid(self.feature_reduction(out)) 
        return self.fc(out)

class WideResNetStrong(nn.Module):
    """Standard WideResNet classifier without the 8-d latent bottleneck."""

    def __init__(self, n_classes=10, depth=28, widen_factor=10, drop_rate=0.3):
        super(WideResNetStrong, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.fc = nn.Linear(self.nChannels, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class HybridQWideResNet(nn.Module):  
    def __init__(self, n_classes=10, depth=28, widen_factor=10, drop_rate=0.3, n_qubits=8, n_layers=3):
        super(HybridQWideResNet, self).__init__()
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.feature_reduction = nn.Linear(self.nChannels, n_qubits)
        self.quantum_layer = make_quantum_layer(n_qubits, n_layers)
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(-1, self.nChannels)
        out = torch.sigmoid(self.feature_reduction(out)) * (2 * np.pi)
        out = self.quantum_layer(out)
        return self.fc(out)

class HybridQWideResNetNoQuantum(nn.Module):
    """Hybrid-shaped WRN ablation that removes the quantum layer."""

    def __init__(self, n_classes=10, depth=28, widen_factor=10, drop_rate=0.3,
                 n_qubits=8, n_layers=3):
        super(HybridQWideResNetNoQuantum, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.feature_reduction = nn.Linear(self.nChannels, n_qubits)
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(-1, self.nChannels)
        out = torch.sigmoid(self.feature_reduction(out)) * (2 * np.pi)
        return self.fc(out)

class HybridQWideResNetMLP(nn.Module):
    """Hybrid-shaped WRN ablation that replaces the quantum layer with an MLP."""

    def __init__(self, n_classes=10, depth=28, widen_factor=10, drop_rate=0.3,
                 n_qubits=8, n_layers=3):
        super(HybridQWideResNetMLP, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.feature_reduction = nn.Linear(self.nChannels, n_qubits)
        self.classical_layer = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.ReLU(),
            nn.Linear(n_qubits, n_qubits),
            nn.ReLU(),
        )
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(-1, self.nChannels)
        out = torch.sigmoid(self.feature_reduction(out)) * (2 * np.pi)
        out = self.classical_layer(out)
        return self.fc(out)
