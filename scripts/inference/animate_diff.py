# Read video

import torch
import torch.nn as nn
from einops import rearrange
from decord import VideoReader
import time

# Original InflatedConv3d
class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        return x

# Depthwise Separable InflatedConv3d
class DepthwiseSeparableInflatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = InflatedConv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = InflatedConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# Read video
video_path = "output_01.mp4"
video_reader = VideoReader(video_path)
video = video_reader[:].asnumpy()
video = torch.from_numpy(video).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0

# Resize video to match SD sizes
target_height = 256
target_width = 256
video = nn.functional.interpolate(video, size=(video.shape[2], target_height, target_width), mode='trilinear', align_corners=False)

# Set up models
sd_config = {
    "in_channels": 3,
    "out_channels": 3,
    "block_out_channels": (320, 640, 1280, 1280),
    "down_block_types": ("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
    "up_block_types": ("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
}

# original_layers = [InflatedConv3d(sd_config["in_channels"], sd_config["block_out_channels"][0], kernel_size=3, stride=2, padding=1)]
# for i in range(1, len(sd_config["down_block_types"])):
#     original_layers.append(InflatedConv3d(sd_config["block_out_channels"][i-1], sd_config["block_out_channels"][i], kernel_size=3, stride=2, padding=1))
# original_model = nn.Sequential(*original_layers)

optimized_layers = [DepthwiseSeparableInflatedConv3d(sd_config["in_channels"], sd_config["block_out_channels"][0], kernel_size=3, stride=2, padding=1)]
for i in range(1, len(sd_config["down_block_types"])):
    optimized_layers.append(DepthwiseSeparableInflatedConv3d(sd_config["block_out_channels"][i-1], sd_config["block_out_channels"][i], kernel_size=3, stride=2, padding=1))
optimized_model = nn.Sequential(*optimized_layers)

# # Encode with original AnimateDiff
# start_time = time.time()
# original_output = original_model(video)
# original_time = time.time() - start_time
# print(f"Original AnimateDiff time: {original_time:.4f} seconds")

# Encode with optimized AnimateDiff
start_time = time.time()
optimized_output = optimized_model(video)
optimized_time = time.time() - start_time
print(f"Optimized AnimateDiff time: {optimized_time:.4f} seconds")

# # Print output shapes
# print(f"Original output shape: {original_output.shape}")
# print(f"Optimized output shape: {optimized_output.shape}")

# # Save original model checkpoint
# torch.save(original_model.state_dict(), "original_model_checkpoint.pth")

# # Save optimized model checkpoint
# torch.save(optimized_model.state_dict(), "optimized_model_checkpoint.pth")