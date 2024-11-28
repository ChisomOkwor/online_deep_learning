import argparse
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# def weighted_loss(pred, target, mask, alpha=0.7):
#     lateral_error = torch.mean(torch.abs(pred[mask][:, 1] - target[mask][:, 1]))
#     longitudinal_error = torch.mean(torch.abs(pred[mask][:, 0] - target[mask][:, 0]))
#     return alpha * lateral_error + (1 - alpha) * longitudinal_error


# def train_model(
#     model_name: str,
#     transform_pipeline: str,
#     num_workers: int,
#     lr: float,
#     batch_size: int,
#     num_epochs: int,
#     debug: bool = False,
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     writer = SummaryWriter(log_dir="runs/mlp_planner")

#     # Load dataset
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
#         shuffle=True,
#         num_workers=num_workers,
#     )

#     # Load model
#     print(f"Loading model: {model_name}")
#     model = load_model(model_name).to(device)

#     # Optimizer, scheduler, loss
#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
#     criterion = nn.L1Loss()

#     best_lateral_error = float("inf")
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()

#             track_left, track_right, waypoints, mask = (
#                 batch["track_left"].to(device),
#                 batch["track_right"].to(device),
#                 batch["waypoints"].to(device),
#                 batch["waypoints_mask"].to(device),
#             )

#             predictions = model(track_left, track_right)
#             loss = criterion(predictions[mask], waypoints[mask])
#             loss.backward()

#             clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)

#         # Validation
#         model.eval()
#         val_loss, lateral_error, longitudinal_error = 0, 0, 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 track_left, track_right, waypoints, mask = (
#                     batch["track_left"].to(device),
#                     batch["track_right"].to(device),
#                     batch["waypoints"].to(device),
#                     batch["waypoints_mask"].to(device),
#                 )

#                 predictions = model(track_left, track_right)
#                 loss = criterion(predictions[mask], waypoints[mask])
#                 val_loss += loss.item()

#                 lateral_error = torch.mean(torch.abs(predictions[mask][:, 1] - waypoints[mask][:, 1])).item()
#                 longitudinal_error = torch.mean(torch.abs(predictions[mask][:, 0] - waypoints[mask][:, 0])).item()


#         val_loss /= len(val_loader)
#         lateral_error /= len(val_loader)
#         longitudinal_error /= len(val_loader)

#         # Log metrics
#         writer.add_scalar("Loss/Train", train_loss, epoch)
#         writer.add_scalar("Loss/Validation", val_loss, epoch)
#         writer.add_scalar("Error/Lateral", lateral_error, epoch)
#         writer.add_scalar("Error/Longitudinal", longitudinal_error, epoch)

#         print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
#               f"Val Loss: {val_loss:.4f}, Lateral Error: {lateral_error:.4f}, Longitudinal Error: {longitudinal_error:.4f}")

#         if lateral_error < best_lateral_error:
#             best_lateral_error = lateral_error
#             torch.save(model.state_dict(), "best_model.pt")
#             print(f"Saved new best model with lateral error: {best_lateral_error:.4f}")

#         scheduler.step(val_loss)

#     writer.close()
#     print("Training complete. Best lateral error:", best_lateral_error)
#     save_model(model)
#     print(f"Model {model_name} saved successfully!")
    


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a planner model")
#     parser.add_argument("--model", type=str, required=True, choices=["mlp_planner"], help="Model to train")
#     parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
#     parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
#     parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed outputs")
#     args = parser.parse_args()

#     train_model(
#         args
#     )


def weighted_loss(pred, target, mask, alpha=0.7):
    # Ensure the mask applies to the batch and waypoints dimensions
    mask = mask.unsqueeze(-1).expand_as(pred)  # Expand mask to match pred and target

    # Apply the mask
    pred_masked = pred[mask].view(-1, 2)  # Reshape to (-1, 2) after masking
    target_masked = target[mask].view(-1, 2)

    # Calculate errors
    lateral_error = torch.mean(torch.abs(pred_masked[:, 1] - target_masked[:, 1]))
    longitudinal_error = torch.mean(torch.abs(pred_masked[:, 0] - target_masked[:, 0]))

    # Weighted combination of errors
    return alpha * lateral_error + (1 - alpha) * longitudinal_error



def train_model(
    model_name: str,
    transform_pipeline: str,
    num_workers: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    debug: bool = False,
    grad_clip=5.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/mlp_planner")

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

    # Use L1 Loss
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_lateral_error = float('inf')  # Keep track of the best lateral error
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
            loss = weighted_loss(predictions[mask], waypoints[mask], mask)  # Apply mask
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), grad_clip)
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

                #Long error has to be 
                
                longitudinal_error += torch.mean(torch.abs(predictions[mask][:, 0] - waypoints[mask][:, 0])).item()
                lateral_error += torch.mean(torch.abs(predictions[mask][:, 1] - waypoints[mask][:, 1])).item()

        val_loss /= len(val_loader)
        lateral_error /= len(val_loader)
        longitudinal_error /= len(val_loader)


         # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Error/Lateral", lateral_error, epoch)
        writer.add_scalar("Error/Longitudinal", longitudinal_error, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Lateral Error: {lateral_error:.4f}, Longitudinal Error: {longitudinal_error:.4f}")

        # Save the model with the best lateral error
        if lateral_error < 0.5 and longitudinal_error < 0.2:
            best_lateral_error = lateral_error
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved with lateral error: {best_lateral_error:.4f}")
            save_model(model)

        # Adjust learning rate
        scheduler.step(val_loss)

    
    # Close the writer and save the model
    writer.close()
    print("Training complete. Best lateral error:", best_lateral_error)
    # save_model(model)
    # print(f"Model {model_name} saved successfully!")
