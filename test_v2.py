"""Evaluation helper to quickly run the test split and visualize some predictions.

This script loads the saved checkpoint and computes test accuracy then displays a
small grid of example predictions using matplotlib.
"""

import torch
from model_v2 import BetterCNN
from preprocess_v2 import get_loaders
import matplotlib.pyplot as plt


def test_v2():
    data_dir = "dataset/DATASET"
    batch_size = 32

    _, _, test_loader, classes = get_loaders(data_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterCNN(num_classes=len(classes)).to(device)

    state_dict = torch.load("model_v2/best_model_v2.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"🧪 Test Accuracy (V2 Model): {accuracy:.2f}% on {total} images")
    print("Classes:", classes)

    # Show a few predictions
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images_display = images[:8]
    labels_display = labels[:8]

    outputs = model(images_display.to(device))
    _, preds = torch.max(outputs, 1)

    # Unnormalize for display
    images_display = images_display * 0.5 + 0.5

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = images_display[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Pred: {classes[preds[i]]}\nActual: {classes[labels_display[i]]}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_v2()
