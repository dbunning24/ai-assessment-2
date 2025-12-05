import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from data.dataset import GalaxyZooDataset
from models.simclr import SimCLR
from models.nt_xent import NTXentLoss

from tqdm import tqdm
import argparse
import os


def train(
    csv_path: str,
    mapping_csv: str,
    image_dir: str,
    batch_size: int = 256,
    epochs: int = 100,
    n_samples: int = 100000,
    lr: float = 1e-3,
    temperature: float = 0.5,
    device: str | None = None,
):
    # -------------------------
    # Device setup
    # -------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Dataset + DataLoader
    # -------------------------
    dataset = GalaxyZooDataset(
        csv_path=csv_path,
        mapping_csv=mapping_csv,
        image_dir=image_dir,
        n_samples=n_samples,
        image_size=64,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device == "cuda" else 0,
        drop_last=True,
    )

    print(f"Loaded {len(dataset)} galaxies for training.")

    # -------------------------
    # Model + Loss + Optimizer
    # -------------------------
    model = SimCLR().to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Training loop
    # -------------------------
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for view1, view2, _ in progress:
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            _, z_i = model(view1)
            _, z_j = model(view2)

            # Contrastive loss
            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")

    # -------------------------
    # Save the encoder weights
    # -------------------------
    save_path = "encoder.pth"
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved encoder weights to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimCLR on Galaxy Zoo 2")

    parser.add_argument("--csv-path", type=str, default="data/gz2spec.csv")
    parser.add_argument("--mapping-csv", type=str, default="data/gz2maps.csv")
    parser.add_argument("--image-dir", type=str, default="data/images")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)

    parser.add_argument("--device", type=str, default=None,
                        help="force 'cpu' or 'cuda'; default auto-detect")

    args = parser.parse_args()

    train(
        csv_path=args.csv_path,
        mapping_csv=args.mapping_csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_samples=args.n_samples,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
    )
