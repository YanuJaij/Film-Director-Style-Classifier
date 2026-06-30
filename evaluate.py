import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

def main():
    DATA_DIR  = "./dataset"
    SAVE_PATH = "./best_model.pth"
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    class_names = val_dataset.classes
    NUM_CLASSES = len(class_names)
    print(f"Classes: {class_names}")

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        mask = all_labels == i
        acc  = (all_preds[mask] == all_labels[mask]).mean() * 100
        print(f"  {name:<25} {acc:.1f}%")

    print("\nFull classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    labels_short = [n.replace("_", "\n") for n in class_names]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_short, yticklabels=labels_short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Director Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion_matrix.png")

if __name__ == "__main__":
    main()