from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# class MLPPlanner(nn.Module):
#     def __init__(self, n_track: int = 10, n_waypoints: int = 3, hidden_dim1: int = 64, hidden_dim2: int = 32):
#         super().__init__()

#         self.n_track = n_track
#         self.n_waypoints = n_waypoints

#         input_dim = 4 * n_track * 2  # track boundaries, centerline, lane width
#         output_dim = n_waypoints * 2  # (x, y) coordinates for waypoints

#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),  # Add dropout for regularization
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim2, output_dim),
#         )

#     def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
#         centerline = (track_left + track_right) / 2
#         lane_width = torch.norm(track_left - track_right, dim=2, keepdim=True)

#         track_left = (track_left - track_left.mean(dim=1, keepdim=True)) / (track_left.std(dim=1, keepdim=True) + 1e-6)
#         track_right = (track_right - track_right.mean(dim=1, keepdim=True)) / (track_right.std(dim=1, keepdim=True) + 1e-6)
#         centerline = (centerline - centerline.mean(dim=1, keepdim=True)) / (centerline.std(dim=1, keepdim=True) + 1e-6)
#         lane_width = (lane_width - lane_width.mean(dim=1, keepdim=True)) / (lane_width.std(dim=1, keepdim=True) + 1e-6)

#         lane_width = lane_width.repeat(1, 1, 2)
#         x = torch.cat([track_left, track_right, centerline, lane_width], dim=1).view(track_left.shape[0], -1)
#         return self.mlp(x).view(track_left.shape[0], self.n_waypoints, 2)


class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, hidden_dim: int = 32):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 4 * n_track * 2  # track boundaries, centerline, lane width
        output_dim = n_waypoints * 2  # (x, y) coordinates for waypoints

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        centerline = (track_left + track_right) / 2
        lane_width = torch.norm(track_left - track_right, dim=2, keepdim=True)

        # Normalize inputs
        track_left = (track_left - track_left.mean(dim=1, keepdim=True)) / (track_left.std(dim=1, keepdim=True) + 1e-6)
        track_right = (track_right - track_right.mean(dim=1, keepdim=True)) / (track_right.std(dim=1, keepdim=True) + 1e-6)
        centerline = (centerline - centerline.mean(dim=1, keepdim=True)) / (centerline.std(dim=1, keepdim=True) + 1e-6)
        lane_width = (lane_width - lane_width.mean(dim=1, keepdim=True)) / (lane_width.std(dim=1, keepdim=True) + 1e-6)

        lane_width = lane_width.repeat(1, 1, 2)
        x = torch.cat([track_left, track_right, centerline, lane_width], dim=1).view(track_left.shape[0], -1)
        return self.mlp(x).view(track_left.shape[0], self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_track (int): Number of points in each side of the track
            n_waypoints (int): Number of waypoints to predict
            d_model (int): Dimension of the transformer model
            num_layers (int): Number of layers in the Transformer decoder
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input encoding layer: projects (x, y) coordinates to d_model
        self.input_encoder = nn.Linear(2, d_model)

        # Learned query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection layer: projects d_model back to (x, y) coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (B, n_track, 2)
            track_right (torch.Tensor): shape (B, n_track, 2)

        Returns:
            torch.Tensor: Predicted waypoints with shape (B, n_waypoints, 2)
        """
        # Concatenate left and right track boundaries
        x = torch.cat([track_left, track_right], dim=1)  # Shape: (B, 2 * n_track, 2)

        # Encode input features
        x = self.input_encoder(x)  # Shape: (B, 2 * n_track, d_model)

        # Get waypoint queries
        query = self.query_embed.weight.unsqueeze(0).repeat(x.size(0), 1, 1)  # Shape: (B, n_waypoints, d_model)

        # Pass through TransformerDecoder
        out = self.decoder(query, x)  # Shape: (B, n_waypoints, d_model)

        # Project output embeddings to (x, y) coordinates
        waypoints = self.output_proj(out)  # Shape: (B, n_waypoints, 2)

        return waypoints



class CNNPlanner(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
    ):
        """
        Args:
            n_waypoints (int): Number of waypoints to predict
            hidden_dim (int): Size of the hidden layer in the fully connected network
        """
        super().__init__()

        self.n_waypoints = n_waypoints

        # Input normalization
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 48, 64)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 24, 32)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 12, 16)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 12 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waypoints * 2),  # Predict (x, y) for each waypoint
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predicts waypoints from the input image.

        Args:
            image (torch.FloatTensor): shape (B, 3, 96, 128) with values in [0, 1]

        Returns:
            torch.FloatTensor: Predicted waypoints with shape (B, n_waypoints, 2)
        """
        # Normalize input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Extract features using CNN
        x = self.cnn(x)  # Shape: (B, 128, 12, 16)

        # Flatten features
        x = x.view(x.size(0), -1)  # Shape: (B, 128 * 12 * 16)

        # Predict waypoints
        x = self.fc(x)  # Shape: (B, n_waypoints * 2)

        # Reshape to (B, n_waypoints, 2)
        waypoints = x.view(x.size(0), self.n_waypoints, 2)

        return waypoints



MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
