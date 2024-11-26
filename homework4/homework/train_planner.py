import argparse
import torch
from torch.utils.data import DataLoader
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(
    model_name: str,
    transform_pipeline: str = "state_only",
    num_workers: int = 4,
    lr: float = 1e-4,
    batch_size: int = 128,
    num_epoch: int = 100,
):
    print(f"Training {model_name} with lr={lr}, batch_size={batch_size}, num_epoch={num_epoch}")

    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset...")
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Load model
    print(f"Loading model: {model_name}")
    # model = load_model(model_name).to(device)
    model = load_model(model_name)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    print("Starting training...")
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Prepare inputs based on model type
            if model_name == "cnn_planner":
                # CNNPlanner expects images as input
                # inputs = batch["image"].to(device)
                inputs = batch["image"]

                pred_waypoints = model(inputs)
            else:
                # Other planners expect track boundaries as input
                track_left = batch["track_left"]
                track_right = batch["track_right"]

                # track_left = batch["track_left"].to(device)
                # track_right = batch["track_right"].to(device)
                pred_waypoints = model(track_left=track_left, track_right=track_right)

            # Get ground truth waypoints
            # waypoints = batch["waypoints"].to(device)
            waypoints = batch["waypoints"]


            # Compute loss
            loss = criterion(pred_waypoints, waypoints)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epoch} - Training Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "cnn_planner":
                    # inputs = batch["image"].to(device)

                    inputs = batch["image"]
                    pred_waypoints = model(inputs)
                else:
                    # track_left = batch["track_left"].to(device)
                    # track_right = batch["track_right"].to(device)
                    track_left = batch["track_left"]
                    track_right = batch["track_right"]
                    pred_waypoints = model(track_left=track_left, track_right=track_right)
                
                # waypoints = batch["waypoints"].to(device)
                waypoints = batch["waypoints"]
                loss = criterion(pred_waypoints, waypoints)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epoch} - Validation Loss: {val_loss:.4f}")

    # Save model
    save_model(model)
    print(f"Model {model_name} saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"], help="Model to train")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--validate", action="store_true", help="Enable validation during training")
    args = parser.parse_args()

    train(args)
