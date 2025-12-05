import torch
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import GalaxyZooDataset
from models.simclr import SimCLR


def extract_features(
    csv_path: str,
    mapping_csv: str,
    image_dir: str,
    batch_size: int = 256,
    n_samples: int = 100000,
    device: str | None = None,
):
    # -------------------------
    # Device setup
    # -------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Dataset + Loader
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
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
    )

    print(f"Extracting features for {len(dataset)} galaxies...")

    # -------------------------
    # Load encoder
    # -------------------------
    model = SimCLR().to(device)
    model.encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    model.eval()

    all_features = []
    all_ids = []

    # -------------------------
    # Feature extraction loop
    # -------------------------
    with torch.no_grad():
        for view1, view2, galaxy_id in dataloader:
            view1 = view1.to(device)

            # only encode one view — both are same galaxy
            h, _ = model(view1)

            all_features.append(h.cpu().numpy())
            all_ids.append(galaxy_id.numpy())

    # -------------------------
    # Save outputs
    # -------------------------
    all_features = np.concatenate(all_features, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    np.save("features.npy", all_features)
    np.save("ids.npy", all_ids)

    print("Saved features to features.npy")
    print("Saved ids to ids.npy")


if __name__ == "__main__":
    extract_features(
        csv_path="data/gz2spec.csv",
        mapping_csv="data/gz2maps.csv",
        image_dir="data/images/",
    )
