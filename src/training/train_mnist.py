import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import copy
from src.models.complex_ffnn import ComplexFFNN
from src.optimizers.custom_adam_adaptive_momentum import custom_adam_adaptive_momentum

# ---------------------
# Config
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 5
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.0
seed = 42
torch.manual_seed(seed)

# ---------------------
# Data
# ---------------------
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_size = 5000
train_size = len(train_ds) - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,      batch_size=256, shuffle=False)

# ---------------------
# Training Function
# ---------------------
def train_with_accuracy(model, optimizer_type='torch', epochs=5):
    criterion = nn.CrossEntropyLoss()
    state = {}
    batch_losses, batch_train_acc, val_acc_per_epoch = [], [], []

    if optimizer_type == 'torch':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        correct_batch, total_batch = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            if optimizer_type == 'torch':
                optimizer.step()
            elif optimizer_type == 'custom':
                params = [p for p in model.parameters() if p.grad is not None]
                grads  = [p.grad for p in params]
                custom_adam_adaptive_momentum(params, grads, state, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=0.5)
            else:
                raise ValueError("Invalid optimizer_type")

            batch_losses.append(loss.item())

            preds = out.argmax(dim=1)
            correct_batch += (preds == yb).sum().item()
            total_batch   += yb.size(0)
            batch_train_acc.append(correct_batch / total_batch)

        # Validation accuracy
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct_val += (pred == yb).sum().item()
                total_val   += yb.size(0)
        val_acc_per_epoch.append(correct_val / total_val)

    return batch_losses, batch_train_acc, val_acc_per_epoch

# ---------------------
# Setup models
# ---------------------
base_model = ComplexFFNN().to(device)
model_torch  = copy.deepcopy(base_model).to(device)
model_custom = copy.deepcopy(base_model).to(device)

# ---------------------
# Train models
# ---------------------
loss_torch, train_acc_torch, val_acc_torch   = train_with_accuracy(model_torch,  'torch',  epochs=epochs)
loss_custom, train_acc_custom, val_acc_custom = train_with_accuracy(model_custom, 'custom', epochs=epochs)

# ---------------------
# Plots
# ---------------------
plt.figure(figsize=(10,6))
plt.plot(loss_torch, label='Torch Adam')
plt.plot(loss_custom, label='Custom Adaptive Momentum Adam')
plt.xlabel('Batch index'); plt.ylabel('Loss'); plt.title('Batch-wise Training Loss')
plt.legend(); plt.grid(True); plt.savefig('figures/loss_compare.png', dpi=150)

plt.figure(figsize=(10,6))
plt.plot(train_acc_torch, label='Torch Adam Train Acc')
plt.plot(train_acc_custom, label='Custom Adaptive Momentum Adam Train Acc')
plt.xlabel('Batch index'); plt.ylabel('Train Accuracy'); plt.title('Training Accuracy per Batch')
plt.legend(); plt.grid(True); plt.savefig('figures/train_acc_compare.png', dpi=150)

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), val_acc_torch, marker='o', label='Torch Adam Val Acc')
plt.plot(range(1, epochs+1), val_acc_custom, marker='o', label='Custom Adaptive Momentum Adam Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy'); plt.title('Validation Accuracy per Epoch')
plt.legend(); plt.grid(True); plt.savefig('figures/val_acc_compare.png', dpi=150)
plt.show()

# ---------------------
# Test evaluation
# ---------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    return correct / total

acc_torch  = evaluate(model_torch,  test_loader)
acc_custom = evaluate(model_custom, test_loader)
print(f"Test Accuracy — Torch Adam: {acc_torch:.4f}, Custom Adaptive Momentum Adam: {acc_custom:.4f}")
