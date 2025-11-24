"""Training script for the V2 model.

This module provides a simple `train_v2()` function that runs training and validation
loops and saves the best model to `model_v2/best_model_v2.pth`.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model_v2 import BetterCNN
from preprocess_v2 import get_loaders


def train_v2():
    """Train the V2 CNN and save the best-performing checkpoint.

    The function uses a ReduceLROnPlateau scheduler and prints progress to stdout.
    """
    data_dir = "dataset/DATASET"
    batch_size = 128
    epochs = 12

    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterCNN(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Reduce LR when val accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_val_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    print("\n🚀 Starting Training V2...\n")
    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"- Train Loss: {avg_train_loss:.4f} "
              f"- Val Loss: {avg_val_loss:.4f} "
              f"- Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("model_v2", exist_ok=True)
            torch.save(model.state_dict(), "model_v2/best_model_v2.pth")
            print(f"💾 New best model saved with Val Acc: {best_val_acc:.2f}%")

    # Plot curves (optional visualizations)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves (V2)")
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy (V2)")
    plt.show()

    print(f"\n✅ Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train_v2()
