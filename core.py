import os
from copy import deepcopy
from scipy.signal import stft
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from daspy import Section

def same_padding(arr, axis, target_length):
    """
    Pad a NumPy array along a specified axis to a target length, using the first and last slices
    of the axis for left/right padding (pads equally on both ends, with right padding getting the extra if odd).

    Parameters:
        arr (np.ndarray): Input array to pad.
        axis (int): Axis along which to pad (e.g., 0 for rows, 1 for columns).
        target_length (int): Desired length of the array along the target axis.

    Returns:
        np.ndarray: Padded array with the target length on the specified axis.

    Raises:
        ValueError: If target_length is less than the current length of the axis.
    """
    # Get current length of the target axis
    current_length = arr.shape[axis]
    
    # Check if padding is needed
    if target_length < current_length:
        raise ValueError(f"Target length ({target_length}) must be >= current length ({current_length})")
    elif target_length == current_length:
        return arr.copy()  # No padding needed
    
    # Calculate total padding and split into left (before) and right (after)
    total_pad = target_length - current_length
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    # Create slice objects to extract the first/last element(s) of the target axis
    # Example: For axis=0, first_slice is arr[0:1, ...], last_slice is arr[-1:, ...]
    slices = [slice(None)] * arr.ndim  # Slice all axes by default
    slices[axis] = slice(0, 1)  # First slice of the target axis
    first_slice = arr[tuple(slices)]
    
    slices[axis] = slice(-1, None)  # Last slice of the target axis
    last_slice = arr[tuple(slices)]

    # Repeat the first/last slice to match the padding lengths along the target axis
    # Use np.repeat to tile along the target axis (other axes remain as-is)
    pad_left_arr = np.repeat(first_slice, pad_left, axis=axis)
    pad_right_arr = np.repeat(last_slice, pad_right, axis=axis)

    # Concatenate the padding arrays with the original array along the target axis
    padded_arr = np.concatenate([pad_left_arr, arr, pad_right_arr], axis=axis)

    return padded_arr

def clip_norm(rawdata):
    data = deepcopy(rawdata)
    if np.iscomplexobj(data):
        data = abs(data)
        d0 = np.percentile(data, 5)
        d1 = np.percentile(data, 95)
        return (np.clip(data, d0, d1) - d0) / (d1 - d0)
    
    d1 = np.percentile(abs(data), 90)
    return np.clip(data, -d1, d1) / d1

def sliding_window(rawdata: NDArray, dim, window_size, window_step):
    """
    Padding both ends using the first and last slices.
    """
    current_length = rawdata.shape[dim]
    if current_length < window_size:
        total_length = window_size
    elif (current_length - window_size) % window_step != 0:
        total_length = (((current_length - window_size) // window_step) + 1) * window_step + window_size
    else:
        total_length = current_length
    data = same_padding(rawdata, dim, total_length)
    rawindexes = np.arange(current_length)
    indexes = same_padding(rawindexes, 0, total_length)
    starts = indexes[::window_step][:-1]
    slice_list = []
    for start in np.arange(0, total_length, window_step):
        if start + window_size > total_length:
            break
        slice_list.append(np.take(data, range(start, start + window_size), axis = dim))
    
    return starts, np.stack(slice_list)

def customized_stft(sec: Section, nperseg = 160, noverlap = 80, channels_per_window = 64):
    _, start_times, stft_mat = stft(sec.data, fs = sec.fs, nperseg = nperseg, noverlap = noverlap, boundary = None)
    stft_normed = clip_norm(stft_mat)[:, :64, :] # (channel, freq, time)
    start_channels, stft_splits = sliding_window(stft_normed.transpose(2, 0, 1), dim = 1, 
                                                    window_size = channels_per_window, window_step = channels_per_window >> 1,
                                                    )
    stft_splits = stft_splits.swapaxes(0, 1)
    return start_times, start_channels, stft_splits

def get_encoder(latent_dim: int) -> nn.Module:
    model = ConvAutoEncoder(latent_dim=latent_dim)
    checkpoint = torch.load(os.path.join("models", f'autoencoder_64x64_latent{latent_dim}.pth'), 
                            map_location=torch.device('cpu'), weights_only = True)
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder = Encoder(model.encoder)
    encoder.to("cuda")
    encoder.eval()
    return encoder

class Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x):
        dim0, dim1, dim2, dim3 = x.shape
        flattened = x.reshape(dim0 * dim1, 1, dim2, dim3)
        output = self.encoder(flattened)
        return output.reshape(dim0, dim1, -1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTransposeBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.deconv(x)

class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoEncoder, self).__init__()
        channels = [1, 32, 64, 128, 256]
        # channels = [1, 16, 32, 64, 128]
        encoder = nn.ModuleList(ConvBlock(channels[i], channels[i+1]) for i in range(len(channels)-1))
        decoder = nn.ModuleList(ConvTransposeBlock(channels[i+1], channels[i]) for i in range(len(channels)-2, 0, -1))
        # Encoder
        self.encoder = nn.Sequential(
            *encoder,
            
            nn.Flatten(),
            nn.Linear(channels[-1] * 4 * 4, latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, channels[-1] * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (channels[-1], 4, 4)),
            
            *decoder,

            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FeaturesFC(nn.Module):
    def __init__(self, latent_dim = 32, num_classes = 6):
        super(FeaturesFC, self).__init__()
        hidden_dim = 64
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
        self.network2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out = self.network(x)
        out = out.squeeze()
        return self.network2(out)