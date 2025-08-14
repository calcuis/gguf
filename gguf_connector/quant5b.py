import torch
def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)
def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1
        )
def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 63, m_d & 15 | d >> 2 & 48], dim=-1)
    min = torch.cat([m & 63, m_d >> 4 | m >> 2 & 48], dim=-1)
    return sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8))
from math import ceil, log2
def load_grid_tensor(grid_shape, grid_hex, grid_map, device):
    bits_per_elem = ceil(log2(len(grid_map)))
    elems_per_byte = 8 // bits_per_elem
    grid_bytes = torch.tensor(list(grid_hex), dtype=torch.uint8, device=device)
    grid = grid_bytes.view(-1, 2)
    mask = grid > 64
    grid = torch.where(mask, grid + 9, grid) & 15
    shifts = torch.tensor([4, 0], dtype=torch.uint8, device=device).view(1, 2)
    grid = (grid << shifts).sum(dim=1)
    shift_vals = torch.arange(0, 8, 8 // elems_per_byte, device=device).view(
        1, elems_per_byte)
    grid = grid.view(-1, 1) >> shift_vals & (1 << bits_per_elem) - 1
    grid_map_tensor = torch.tensor(grid_map, dtype=torch.float32, device=device
        ).view(1, -1)
    grid = torch.take_along_dim(grid_map_tensor, grid, dim=1)
    return grid.view(1, 1, *grid_shape)
def e8m0_to_fp32_half(x):
    x = x.to(torch.int32)
    bits = torch.where(x < 2, torch.int32(2097152) << x, x - 1 << torch.
        int32(23))
    return bits.view(torch.float32)