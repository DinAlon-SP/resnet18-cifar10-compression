import os
import time
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils.model_utils import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_set, val_set, test_set,
          run_name="resnet_run1",
          epochs=5, lr=0.001,
          optimizer_name='Adam',
          loss_name='CrossEntropy',
          batch_size=128,
          use_scheduler=True,
          label_smoothing=0,
          weight_decay=0.0,
          l1_lambda=0.0,
          momentum=0.0,
          amp_mode='none'):  # 'none', 'full'


    # === Output paths ===
    script_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(script_path))
    results_dir = os.path.join(parent_dir, "results")
    output_dir = os.path.join(results_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path  = os.path.join(output_dir, "best_model.pth")
    plot_path   = os.path.join(output_dir, "loss_plot.png")
    config_path = os.path.join(output_dir, "config.json")
    log_path    = os.path.join(output_dir, "training_log.txt")

    # === Logging ===
    import sys
    class Logger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger(log_path)

    # === Dataloaders ===
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # === Optimizer and Loss ===
    def get_optimizer(name, params):
        if name == 'SGD':
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        return {
            'Adam':     optim.Adam,
            'AdamW':    optim.AdamW,
            'RMSprop':  optim.RMSprop,
            'Adagrad':  optim.Adagrad,
            'NAdam':    optim.NAdam,
            'Adadelta': optim.Adadelta
        }.get(name, optim.Adam)(params, lr=lr, weight_decay=weight_decay)

    losses = {
        'CrossEntropy': lambda: nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        'MSE':          nn.MSELoss,
        'L1':           nn.L1Loss,
        'SmoothL1':     nn.SmoothL1Loss,
        'NLL':          nn.NLLLoss,
        'KLDiv':        nn.KLDivLoss
    }

    optimizer = get_optimizer(optimizer_name, model.parameters())
    criterion = losses.get(loss_name, lambda: nn.CrossEntropyLoss())()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None
    scaler = GradScaler() if amp_mode != 'none' else None

    best_model = copy.deepcopy(model)
    best_val_loss = float("inf")
    train_losses, val_losses, epoch_times = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        start = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if amp_mode == 'full':
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if l1_lambda > 0:
                        loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if l1_lambda > 0:
                    loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if amp_mode == 'full':
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if l1_lambda > 0:
                            loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if l1_lambda > 0:
                        loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), model_path)
            print(f" New best model saved! Val Loss: {best_val_loss:.2f}")
            print(f" Model saved to: {model_path}")

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        print(f" Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"⏱️ Epoch Time: {epoch_time:.2f}s")

    # === Final Test Accuracy ===
    test_acc = evaluate(best_model, test_loader)
    avg_time = sum(epoch_times) / len(epoch_times)
    print(f" Test Accuracy of Best Model: {test_acc:.2f}%")
    print(f" Avg Epoch Time: {avg_time:.2f} sec")

    # === Save Loss Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f" Loss plot saved to: {plot_path}")

    # === Save Config ===
    config = {
        "epochs": epochs,
        "lr": lr,
        "optimizer_name": optimizer_name,
        "loss_name": loss_name,
        "batch_size": batch_size,
        "use_scheduler": use_scheduler,
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
        "l1_lambda": l1_lambda,
        "momentum": momentum,
        "amp_mode": amp_mode
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return best_model, avg_time, train_losses, val_losses
