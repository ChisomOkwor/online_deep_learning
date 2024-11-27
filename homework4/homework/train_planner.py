import argparse
import torch
from torch.utils.data import DataLoader
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Custom weighted loss function
def weighted_mse_loss(pred, target):
    """
    Custom weighted MSE loss to prioritize lateral error.
    Args:
        pred: Predicted waypoints (batch_size, n_waypoints, 2)
        target: Ground truth waypoints (batch_size, n_waypoints, 2)
    Returns:
        Weighted loss: 0.3 * longitudinal_error + 1.7 * lateral_error
    """
    # Lateral error: Difference in x-coordinates
    lateral_error = torch.mean((pred[..., 0] - target[..., 0]) ** 2)

    # Longitudinal error: Difference in y-coordinates
    longitudinal_error = torch.mean((pred[..., 1] - target[..., 1]) ** 2)

    # Weighted loss
    return 0.3 * longitudinal_error + 1.7 * lateral_error


def train(
    model_name: str,
    transform_pipeline: str = "state_only",
    num_workers: int = 4,
    lr: float = 1e-3,  # Updated learning rate
    batch_size: int = 256,  # Updated batch size
    num_epoch: int = 40,
):
    print(f"Training {model_name} with lr={lr}, batch_size={batch_size}, num_epoch={num_epoch}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = load_model(model_name).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Decay learning rate
    print(f"Using Adam optimizer with initial lr={lr} and step LR scheduler")

    # Training loop
    print("Starting training...")
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        train_lateral_error = 0.0
        train_longitudinal_error = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epoch}"):
            optimizer.zero_grad()

            # Prepare inputs based on model type
            if model_name == "cnn_planner":
                inputs = batch["image"].to(device)
                pred_waypoints = model(inputs)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                pred_waypoints = model(track_left=track_left, track_right=track_right)

            # Ground truth waypoints
            waypoints = batch["waypoints"].to(device)

            # Compute losses
            lateral_error = torch.mean((pred_waypoints[..., 0] - waypoints[..., 0]) ** 2)
            longitudinal_error = torch.mean((pred_waypoints[..., 1] - waypoints[..., 1]) ** 2)
            loss = weighted_mse_loss(pred_waypoints, waypoints)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            train_lateral_error += lateral_error.item()
            train_longitudinal_error += longitudinal_error.item()

        # Average training metrics
        train_loss /= len(train_loader)
        train_lateral_error /= len(train_loader)
        train_longitudinal_error /= len(train_loader)
        print(
            f"Epoch {epoch + 1}/{num_epoch} - Training Loss: {train_loss:.4f}, "
            f"Lateral Error: {train_lateral_error:.4f}, Longitudinal Error: {train_longitudinal_error:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_lateral_error = 0.0
        val_longitudinal_error = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{num_epoch}"):
                if model_name == "cnn_planner":
                    inputs = batch["image"].to(device)
                    pred_waypoints = model(inputs)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    pred_waypoints = model(track_left=track_left, track_right=track_right)

                waypoints = batch["waypoints"].to(device)

                # Compute losses
                lateral_error = torch.mean((pred_waypoints[..., 0] - waypoints[..., 0]) ** 2)
                longitudinal_error = torch.mean((pred_waypoints[..., 1] - waypoints[..., 1]) ** 2)
                loss = weighted_mse_loss(pred_waypoints, waypoints)

                # Accumulate metrics
                val_loss += loss.item()
                val_lateral_error += lateral_error.item()
                val_longitudinal_error += longitudinal_error.item()

        # Average validation metrics
        val_loss /= len(val_loader)
        val_lateral_error /= len(val_loader)
        val_longitudinal_error /= len(val_loader)
        print(
            f"Epoch {epoch + 1}/{num_epoch} - Validation Loss: {val_loss:.4f}, "
            f"Lateral Error: {val_lateral_error:.4f}, Longitudinal Error: {val_longitudinal_error:.4f}"
        )

        # Step the scheduler
        scheduler.step()

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

    train(
        model_name=args.model,
        transform_pipeline="state_only",
        num_workers=args.num_workers,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        num_epoch=args.epochs,
    )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a planner model")
#     parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"], help="Model to train")
#     parser.add_argument("--train_data_path", type=str, required=True, help="Path to training dataset")
#     parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation dataset")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
#     parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
#     parser.add_argument("--validate", action="store_true", help="Enable validation during training")
#     args = parser.parse_args()

#     train(args)
