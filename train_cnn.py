#!/usr/bin/env python3
"""
train_cnn.py
Train SimpleCNN on CIFAR-10 using Accelerate CLI (CPU or GPU), PyTorch, and Weights & Biases.

Usage:
  accelerate launch train_cnn.py

Configure the device (CPU/GPU) using:
  accelerate config
  accelerate launch --config_file [your_config.yaml] train_cnn.py
"""

# ---- Imports ----
import os, time, json, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import pandas as pd
from accelerate import Accelerator
import wandb

# ---------- Hyper-parameters ----------
EPOCHS      = 5                  # Number of training epochs
BATCH_SIZE  = 64                 # Mini-batch size
LR          = 1e-3               # Learning rate
PROJECT     = "gpu-vs-cpu-benchmark"  # WandB project name
DATA_DIR    = "./data"          # Directory for downloading CIFAR-10 dataset
# --------------------------------------


# ---- Simple Convolutional Neural Network ----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)          # First convolution layer
        self.pool  = nn.MaxPool2d(2, 2)                      # Max pooling
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)         # Second convolution layer
        self.fc1   = nn.Linear(64 * 16 * 16, 256)            # Fully connected layer
        self.fc2   = nn.Linear(256, 10)                      # Output layer (10 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                # Conv1 + ReLU + Pool
        x = F.relu(self.conv2(x))                           # Conv2 + ReLU
        x = x.flatten(1)                                    # Flatten to feed into FC layer
        x = F.relu(self.fc1(x))                             # FC1 + ReLU
        return self.fc2(x)                                  # FC2 (no softmax; logits used for CrossEntropyLoss)


# ---- Dataloader Preparation ----
def get_dataloaders(bs, accelerator):
    tfm = T.Compose([
        T.ToTensor(),                                      # Convert PIL to tensor
        T.Normalize((0.4914, 0.4822, 0.4465),               # Normalize using CIFAR-10 mean
                    (0.2470, 0.2435, 0.2616))               # and standard deviation
    ])
    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True,
                                            download=True, transform=tfm)
    test_ds  = torchvision.datasets.CIFAR10(DATA_DIR, train=False,
                                            download=True, transform=tfm)

    # Use DataLoader with multiple workers for efficiency
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True)
    test_dl  = torch.utils.data.DataLoader(
        test_ds,  batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True)

    # Prepare dataloaders using accelerator for distributed training
    return accelerator.prepare(train_dl, test_dl)


# ---- Training Loop per Epoch ----
def train_epoch(acc, model, loader, crit, opt):
    model.train()
    running_loss, correct, total = 0., 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(acc.device), yb.to(acc.device)        # Move data to appropriate device
        out  = model(xb)
        loss = crit(out, yb)
        acc.backward(loss)                                   # Backprop using accelerator
        opt.step(); opt.zero_grad()                          # Update weights and reset gradients

        running_loss += loss.item() * xb.size(0)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total   += xb.size(0)
    return running_loss / total, correct / total             # Return average loss and accuracy


# ---- Evaluation (no gradients) ----
@torch.no_grad()
def evaluate(acc, model, loader, crit):
    model.eval()
    running_loss, correct, total = 0., 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(acc.device), yb.to(acc.device)
        out  = model(xb)
        loss = crit(out, yb)
        running_loss += loss.item() * xb.size(0)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total   += xb.size(0)
    return running_loss / total, correct / total             # Return average loss and accuracy


# ---- Main Training Script ----
def main():
    accelerator = Accelerator()                              # Initialize accelerator
    train_dl, test_dl = get_dataloaders(BATCH_SIZE, accelerator)

    model     = SimpleCNN().to(accelerator.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Prepare model, optimizer, and loss for acceleration
    model, optimizer, criterion = accelerator.prepare(model, optimizer, criterion)

    # Log experiment to WandB
    device_str = str(accelerator.device)
    wandb.init(project=PROJECT,
               name=f"cnn-{device_str}",
               tags=[device_str.upper()],
               config=dict(
                   device=device_str, epochs=EPOCHS,
                   batch_size=BATCH_SIZE, lr=LR, model="SimpleCNN"))

    final_val_acc = None
    epoch_times = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(accelerator, model, train_dl, criterion, optimizer)
        val_loss, val_acc = evaluate(accelerator, model, test_dl, criterion)
        dt = time.time() - t0
        epoch_times.append(dt)
        final_val_acc = val_acc

        # Log metrics to WandB for each epoch
        wandb.log(dict(
            epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=val_loss, val_acc=val_acc, epoch_time_sec=dt))

        print(f"[Device={device_str}] Epoch {epoch}/{EPOCHS}: "
              f"train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}, time={dt:.1f}s")

    wandb.finish()  # Close WandB logging

    # ---- Save Model and Metadata ----
    accelerator.wait_for_everyone()                          # Ensure sync across devices
    unwrapped_model = accelerator.unwrap_model(model)        # Remove wrapping for saving
    model_path = f"models/model_{device_str}.pt"
    torch.save(unwrapped_model.state_dict(), model_path)     # Save model weights
    print(f"✅ Saved model to {model_path}")

    # Save training metadata as JSON
    meta = dict(
        device=device_str,
        avg_epoch_time_sec=sum(epoch_times)/len(epoch_times),
        final_val_acc=final_val_acc,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )
    with open(f"models/model_{device_str}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Saved metadata to models/model_{device_str}_meta.json")


# Entry point
if __name__ == "__main__":
    main()
