"""
Data augmentation utilities for brain-to-text decoding.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gauss_smooth(
    inputs: torch.Tensor,
    device: torch.device,
    smooth_kernel_std: float = 2.0,
    smooth_kernel_size: int = 100,
    padding: str = "same",
) -> torch.Tensor:
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data
    along the time axis.

    Args:
        inputs (torch.Tensor): A 3D tensor with shape [B, T, N] (Batch, Time, Features).
        device (torch.device): Device to use for computation.
        smooth_kernel_std (float): Standard deviation of the Gaussian kernel.
        smooth_kernel_size (int): Size of the Gaussian kernel.
        padding (str): Padding mode, either 'same' or 'valid'.

    Returns:
        torch.Tensor: Smoothed 3D tensor with shape [B, T, N].
    """
    # Create Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gauss_kernel = gaussian_filter1d(inp, smooth_kernel_std)
    valid_idx = np.argwhere(gauss_kernel > 0.01)
    gauss_kernel = gauss_kernel[valid_idx]
    gauss_kernel = np.squeeze(gauss_kernel / np.sum(gauss_kernel))

    # Convert to tensor
    gauss_kernel = torch.tensor(gauss_kernel, dtype=inputs.dtype, device=device)
    gauss_kernel = gauss_kernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    batch_size, time_steps, channels = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gauss_kernel = gauss_kernel.repeat(channels, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gauss_kernel, padding=padding, groups=channels)

    return smoothed.permute(0, 2, 1)  # [B, T, C]
