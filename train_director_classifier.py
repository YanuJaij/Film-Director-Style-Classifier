import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def main():
    DATA_DIR    = "./dataset"
    BATCH_SIZE  = 32
    EPOCHS      = 30
    LR          = 1e-3
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH   = "./best_model.pth"

    print(f"Using device: {DEVICE}")

    # augment training frames so the model doesn't just memorise them
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # no augmentation on val - want clean accuracy numbers
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transforms)

    # num_workers=0 because Windows doesn't handle multiprocessing well here
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    NUM_CLASSES = len(train_dataset.classes)
    print(f"Classes ({NUM_CLASSES}): {train_dataset.classes}")

    # pretrained ResNet-18 - freeze everything except the last block and classifier
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += images.size(0)
        return total_loss / total, correct / total

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = run_epoch(train_loader, train=True)
        v_loss, v_acc = run_epoch(val_loader,   train=False)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc*100:.1f}% | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc*100:.1f}%"
              + (" <- best" if v_acc == best_val_acc else ""))

    print(f"\nBest Val Accuracy: {best_val_acc*100:.1f}%  ->  saved to {SAVE_PATH}")

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, EPOCHS + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train")
    ax1.plot(epochs_range, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs_range, [a*100 for a in history["train_acc"]], label="Train")
    ax2.plot(epochs_range, [a*100 for a in history["val_acc"]],   label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")


if __name__ == "__main__":
    main()