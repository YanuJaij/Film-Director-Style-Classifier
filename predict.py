import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


SAVE_PATH = "./best_model.pth"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "david_fincher",
    "denis_villeneuve",
    "guillermo_del_toro",
    "stanley_kubrick",
    "tim_burton",
    "wes_anderson",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze()

    top3 = probs.argsort(descending=True)[:3]

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {CLASS_NAMES[probs.argmax()].replace('_', ' ').title()}")

    print("\nTop 3:")
    for i in top3:
        print(f"  {CLASS_NAMES[i].replace('_', ' ').title():<25} {probs[i]*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    model = load_model()
    for path in sys.argv[1:]:
        predict(path, model)