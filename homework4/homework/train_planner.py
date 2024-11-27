import argparse
import torch
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from tqdm import tqdm


# Custom balanced loss function
def balanced_loss(pred, target):
    """
    Loss function prioritizing lateral error.
    Args:
        pred: Predicted waypoints (batch_size, n_waypoints, 2)
        target: Ground truth waypoints (batch_size, n_waypoints, 2)
    Returns:
        Weighted loss combining lateral and longitudinal errors.
    """
    lateral_error = torch.mean(torch.abs(pred[..., 0] - target[..., 0]))
    longitudinal_error = torch.mean((pred[..., 1] - target[..., 1]) ** 2)
    return 0.6 * lateral_error + 0.4 * longitudinal_error


def train(
    model_name: str,
    transform_pipeline: str = "state_only",
    num_workers: int = 4,
    lr: float = 5e-4,
    batch_size: int = 128,
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

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epoch}"):
            optimizer.zero_grad()

            # Prepare inputs
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)

            # Model prediction
            pred_waypoints = model(track_left=track_left, track_right=track_right)

            # Compute loss
            loss = balanced_loss(pred_waypoints, waypoints)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epoch} - Training Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{num_epoch}"):
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)

                pred_waypoints = model(track_left=track_left, track_right=track_right)
                loss = balanced_loss(pred_waypoints, waypoints)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epoch} - Validation Loss: {val_loss:.4f}")

    # Save model
    save_model(model)
    print(f"Model {model_name} saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner"], help="Model to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
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
