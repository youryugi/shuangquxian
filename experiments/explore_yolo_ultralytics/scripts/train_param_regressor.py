import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class HyperbolaRegressionDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int = 224):
        self.items = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.items.append(json.loads(line))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item["crop_path"]).convert("RGB")
        image_tensor = self.transform(image)
        target = torch.tensor(item["target"], dtype=torch.float32)
        return image_tensor, target


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 5),
        nn.Sigmoid(),
    )
    return model


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)


def main():
    parser = argparse.ArgumentParser(description="Train direct hyperbola-parameter regressor.")
    parser.add_argument("--dataset-dir", type=str, default="../reg_dataset", help="Directory with train.jsonl/val.jsonl")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output-dir", type=str, default="../runs_reg", help="Output dir for checkpoints")
    parser.add_argument("--name", type=str, default="param_regressor", help="Experiment name")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    train_manifest = dataset_dir / "train.jsonl"
    val_manifest = dataset_dir / "val.jsonl"

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("train.jsonl or val.jsonl not found. Run prepare_regression_dataset.py first.")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    train_dataset = HyperbolaRegressionDataset(train_manifest, image_size=args.image_size)
    val_dataset = HyperbolaRegressionDataset(val_manifest, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.output_dir).resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size

        train_loss = running_loss / max(1, count)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "target_order": ["x_vertex_n", "y_vertex_n", "width_n", "height_n", "thickness_n"],
                },
                best_path,
            )

    print("Training finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
