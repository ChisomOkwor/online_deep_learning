import argparse
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from tqdm import tqdm


# # Custom balanced loss function
# def balanced_loss(pred, target):
#     """
#     Loss function prioritizing lateral error.
#     Args:
#         pred: Predicted waypoints (batch_size, n_waypoints, 2)
#         target: Ground truth waypoints (batch_size, n_waypoints, 2)
#     Returns:
#         Weighted loss combining lateral and longitudinal errors.
#     """
#     lateral_error = torch.mean(torch.abs(pred[..., 0] - target[..., 0]))
#     longitudinal_error = torch.mean((pred[..., 1] - target[..., 1]) ** 2)
#     return 0.6 * lateral_error + 0.4 * longitudinal_error

# def dynamic_balanced_loss(pred, target, epoch):
#     # Adjust weights dynamically based on the epoch
#     lateral_weight = 0.7 if epoch < 50 else 0.5
#     longitudinal_weight = 1.0 - lateral_weight
#     lateral_error = torch.mean(torch.abs(pred[..., 0] - target[..., 0]))
#     longitudinal_error = torch.mean((pred[..., 1] - target[..., 1]) ** 2)
#     return lateral_weight * lateral_error + longitudinal_weight * longitudinal_error


# def log_lateral_and_longitudinal_error(pred, target, mask):
#     """
#     Computes lateral and longitudinal errors with masking.
#     Args:
#         pred: Predicted waypoints (batch_size, n_waypoints, 2)
#         target: Ground truth waypoints (batch_size, n_waypoints, 2)
#         mask: Boolean mask for valid waypoints (batch_size, n_waypoints)
#     Returns:
#         Tuple of (lateral_error, longitudinal_error)
#     """
#     error = torch.abs(pred - target)
#     error_masked = error * mask[..., None]

#     lateral_error = error_masked[..., 1].sum() / mask.sum()
#     longitudinal_error = error_masked[..., 0].sum() / mask.sum()

#     return lateral_error.item(), longitudinal_error.item()




# def train(
#     model_name: str,
#     transform_pipeline: str = "state_only",
#     num_workers: int = 4,
#     lr: float = 5e-4,
#     batch_size: int = 128,
#     num_epoch: int = 40,
# ):
#     print("new code is picked up")
#     print(f"Training {model_name} with lr={lr}, batch_size={batch_size}, num_epoch={num_epoch}")

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load dataset
#     print("Loading dataset...")
#     train_loader = load_data(
#         dataset_path="drive_data/train",
#         transform_pipeline=transform_pipeline,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#     )
#     val_loader = load_data(
#         dataset_path="drive_data/val",
#         transform_pipeline=transform_pipeline,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )

#     # Load model
#     print(f"Loading model: {model_name}")
#     model = load_model(model_name).to(device)

#     # Define optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


#     # Training loop
#     print("Starting training...")
#     for epoch in range(num_epoch):
#         model.train()
#         train_loss = 0.0

#         for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epoch}"):
#             optimizer.zero_grad()

#             # Prepare inputs
#             track_left = batch["track_left"].to(device)
#             track_right = batch["track_right"].to(device)
#             waypoints = batch["waypoints"].to(device)

#             # Model prediction
#             pred_waypoints = model(track_left=track_left, track_right=track_right)

#             # Compute loss
#             loss = balanced_loss(pred_waypoints, waypoints)


#             # Backpropagation and optimization
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)
#         print(f"Epoch {epoch + 1}/{num_epoch} - Training Loss: {train_loss:.4f}")

#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         val_lateral_error = 0.0
#         val_longitudinal_error = 0.0

#         with torch.no_grad():
#             for batch in val_loader:
#                 track_left = batch["track_left"].to(device)
#                 track_right = batch["track_right"].to(device)
#                 waypoints = batch["waypoints"].to(device)

#                 pred_waypoints = model(track_left=track_left, track_right=track_right)
#                 torch.save({"pred": pred_waypoints, "target": waypoints, "mask": batch["waypoints_mask"]}, "debug_data.pt")

#                 loss = balanced_loss(pred_waypoints, waypoints)

#                 val_loss += loss.item()

#                 # Log lateral and longitudinal errors
#                 lat_err, long_err = log_lateral_and_longitudinal_error(
#                   pred_waypoints, waypoints, batch["waypoints_mask"].to(device))
#                 val_lateral_error += lat_err
#                 val_longitudinal_error += long_err

#         val_loss /= len(val_loader)
#         val_lateral_error /= len(val_loader)
#         val_longitudinal_error /= len(val_loader)
#         scheduler.step()

#         print(
#             f"Epoch {epoch + 1}/{num_epoch} - Validation Loss: {val_loss:.4f}, "
#             f"Lateral Error: {val_lateral_error:.4f}, Longitudinal Error: {val_longitudinal_error:.4f}"
#         )


#     # Save model
#     save_model(model)
#     print(f"Model {model_name} saved successfully!")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a planner model")
#     parser.add_argument("--model", type=str, required=True, choices=["mlp_planner"], help="Model to train")
#     parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train for")
#     parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
#     args = parser.parse_args()

#     train(
#         model_name=args.model,
#         transform_pipeline="state_only",
#         num_workers=args.num_workers,
#         lr=args.learning_rate,
#         batch_size=args.batch_size,
#         num_epoch=args.epochs,
#     )


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Train a planner model")
# #     parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"], help="Model to train")
# #     parser.add_argument("--train_data_path", type=str, required=True, help="Path to training dataset")
# #     parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation dataset")
# #     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
# #     parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
# #     parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
# #     parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
# #     parser.add_argument("--validate", action="store_true", help="Enable validation during training")
# #     args = parser.parse_args()

# #     train(args)
def train_model(
    model_name: str,
    transform_pipeline: str,
    num_workers: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    debug: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
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

    # Define optimizer, scheduler, and loss
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.L1Loss()

    best_lateral_error = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            track_left, track_right, waypoints, mask = (
                batch["track_left"].to(device),
                batch["track_right"].to(device),
                batch["waypoints"].to(device),
                batch["waypoints_mask"].to(device),
            )

            optimizer.zero_grad()
            predictions = model(track_left, track_right)
            loss = criterion(predictions[mask], waypoints[mask])
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, lateral_error, longitudinal_error = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                track_left, track_right, waypoints, mask = (
                    batch["track_left"].to(device),
                    batch["track_right"].to(device),
                    batch["waypoints"].to(device),
                    batch["waypoints_mask"].to(device),
                )

                predictions = model(track_left, track_right)
                loss = criterion(predictions[mask], waypoints[mask])
                val_loss += loss.item()

                lateral_error += torch.mean(torch.abs(predictions[mask][:, 0] - waypoints[mask][:, 0])).item()
                longitudinal_error += torch.mean(torch.abs(predictions[mask][:, 1] - waypoints[mask][:, 1])).item()

        val_loss /= len(val_loader)
        lateral_error /= len(val_loader)
        longitudinal_error /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Lateral Error: {lateral_error:.4f}, Longitudinal Error: {longitudinal_error:.4f}")

        # Save best model
        if lateral_error < best_lateral_error:
            best_lateral_error = lateral_error
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Saved new best model with lateral error: {best_lateral_error:.4f}")

        scheduler.step(val_loss)

    print("Training complete. Best lateral error:", best_lateral_error)
  # Save model
    save_model(model)
    print(f"Model {model_name} saved successfully!")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner"], help="Model to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed outputs")
    args = parser.parse_args()

    train_model(
        args
    )
