from __future__ import annotations
import math, os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# ------------------------------
# Geometry + wiring helpers
# ------------------------------

def infer_cell_positions_from_arr(arr178: np.ndarray):
    """
    From a 178x178 mosaic grid (>=0 inside circle; -1 outside),
    return:
      pos_cells: (N_cells, 2) float32 [y,x] in 178-space (stable order)
      idx_map  : (178,178) int64, maps [y,x] -> input index or -1
    """
    mask = (arr178 >= 0)
    ys, xs = np.where(mask)
    pos = np.stack([ys, xs], axis=1).astype(np.int32)
    order = np.argsort(ys*1000 + xs)  # stable order
    pos = pos[order]
    idx_map = np.full(arr178.shape, -1, dtype=np.int64)
    for i,(y,x) in enumerate(pos):
        idx_map[y,x] = i
    return pos.astype(np.float32), idx_map   # pos: (N_cells,2) in 178×178 coords


def build_grid_coords_on_178(grid: int, H178: int = 178, W178: int = 178):
    """Return (grid*grid, 2) float32 coordinates (y,x) over 178x178."""
    ys = np.linspace(0, H178-1, grid, dtype=np.float32)
    xs = np.linspace(0, W178-1, grid, dtype=np.float32)
    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([Yg, Xg], axis=-1).reshape(-1, 2)


def out100_to_178_coords(H: int = 100, W: int = 100):
    """
    Center-to-center mapping: 100x100 pixel centers -> 178x178 pixel centers.
    (0 -> 0, 99 -> 177) on each axis.
    """
    s_y = 177.0 / 99.0
    s_x = 177.0 / 99.0
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    y178 = yy * s_y
    x178 = xx * s_x
    return np.stack([y178, x178], axis=-1).reshape(-1, 2)


def make_circle_mask(H: int, W: int, radius: float):
    """Boolean mask for a centered circle in an HxW canvas."""
    cy, cx = (H-1)/2.0, (W-1)/2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return ((yy - cy)**2 + (xx - cx)**2) <= (radius**2)


def neighbor_lists(src_coords: np.ndarray, dst_coords: np.ndarray, radius: float):
    """
    For each dst j, return indices of src within 'radius' (euclidean) of dst[j].
    src_coords: (Ns,2), dst_coords: (Nd,2)
    Output: list of length Nd; each entry np.ndarray[int64] (possibly empty).
    """
    src = torch.from_numpy(src_coords)    # (Ns,2)
    dst = torch.from_numpy(dst_coords)    # (Nd,2)
    lists = []
    chunk = 1024
    for j0 in range(0, dst.shape[0], chunk):
        j1 = min(j0+chunk, dst.shape[0])
        dchunk = dst[j0:j1]               # (C,2)
        diff = dchunk[:, None, :] - src[None, :, :]
        dist2 = (diff*diff).sum(-1)       # (C, Ns)
        within = dist2 <= (radius*radius)
        for row in within:
            idx = torch.nonzero(row, as_tuple=False).squeeze(-1).cpu().numpy()
            lists.append(idx.astype(np.int64))
    return lists


# ------------------------------
# Locally-connected layers
# ------------------------------

class LocalLinear(nn.Module):
    """
    Sparse locally-connected linear layer defined by explicit connection lists.
    Inputs:  (B, N_in)
    Outputs: (B, N_out)
    One weight per (out j, in i) in neighbors[j], plus a bias per output j (if bias=True).
    """
    def __init__(self, N_in: int, neighbors: list[np.ndarray], bias: bool = True, init_scale=0.01):
        super().__init__()
        self.N_in = N_in
        self.N_out = len(neighbors)

        rows, cols = [], []
        for j, idxs in enumerate(neighbors):
            if idxs.size == 0:
                # ensure at least a self-connection to something harmless (will be near zero weight)
                idxs = np.array([0], dtype=np.int64)
            rows.append(np.full(idxs.shape, j, dtype=np.int64))
            cols.append(idxs)
        self.register_buffer("rows", torch.from_numpy(np.concatenate(rows) if rows and rows[0].size else torch.empty(0, dtype=torch.int64).numpy()))
        self.register_buffer("cols", torch.from_numpy(np.concatenate(cols) if cols and cols[0].size else torch.empty(0, dtype=torch.int64).numpy()))

        conn_per_out = np.array([len(ids) for ids in neighbors], dtype=np.float32)
        fan_in_mean  = float(max(1.0, conn_per_out.mean()))

        self.weight = nn.Parameter(torch.randn(self.rows.numel()) * (init_scale / math.sqrt(fan_in_mean)))
        self.bias   = nn.Parameter(torch.zeros(self.N_out)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_in)
        returns: (B, N_out)
        """
        B = x.shape[0]
        out = x.new_zeros(B, self.N_out)
        if self.cols.numel() == 0:
            return out + (self.bias if self.bias is not None else 0)
        gathered = x[:, self.cols]                       # (B, total_conn)
        weighted = gathered * self.weight[None, :]
        for b in range(B):
            out[b].index_add_(0, self.rows, weighted[b])
        if self.bias is not None:
            out += self.bias
        return out


class LocalMLPCircle(nn.Module):
    """
    Input (cells) -> locally-connected hidden grid -> locally-connected output circle (RGB tripled).
    Output activation is sigmoid (predicts [0,1]).
    """
    def __init__(self, N_cells: int, in_to_hidden: list[np.ndarray], hidden_to_out_rgb: list[np.ndarray]):
        super().__init__()
        self.ih = LocalLinear(N_cells, in_to_hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.ho = LocalLinear(len(in_to_hidden), hidden_to_out_rgb, bias=True)
    def forward(self, x):
        h = self.act(self.ih(x))
        y = self.ho(h)
        return torch.sigmoid(y)


# ------------------------------
# Public API
# ------------------------------

def mosaic_to_flat(mosaic_arr: np.ndarray, idx_map: np.ndarray, N_cells: int) -> np.ndarray:
    """
    Convert a 178x178 mosaic (>=0 inside, -1 outside) to a flat (N_cells,) float32 vector.
    """
    if mosaic_arr.shape != (178, 178):
        raise ValueError(f"Expected mosaic (178,178), got {mosaic_arr.shape}")
    vals = mosaic_arr.astype(np.float32).copy()
    vals[vals < 0] = 0.0
    flat = np.zeros((N_cells,), dtype=np.float32)
    ys, xs = np.where(idx_map >= 0)
    flat[idx_map[ys, xs]] = vals[ys, xs]
    return flat


class LocalCirclePredictor:
    """
    End-to-end helper:
      - rebuild wiring from a mosaic mask (arr_178)
      - instantiate LocalMLPCircle
      - load weights from checkpoint
      - predict RGB circle on a 100x100 canvas

    IMPORTANT: hid_grid, r_in, r_out, out_hw, out_radius, and 100->178 mapping
    must match the values used at training.
    """
    def __init__(self,
                 ckpt_path: str,
                 arr_178_for_mask: np.ndarray,
                 hid_grid: int = 400,
                 r_in: float = 6.0,
                 r_out: float = 4.0,
                 out_hw: Tuple[int, int] = (100, 100),
                 out_radius: float = 50.0,
                 device: Optional[torch.device | str] = None):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Inputs (cells)
        pos_cells, idx_map = infer_cell_positions_from_arr(arr_178_for_mask)
        self.pos_cells = pos_cells            # (N_cells,2) float32
        self.idx_map = idx_map                # (178,178) int64
        self.N_cells = pos_cells.shape[0]

        # Hidden grid
        hidden_coords = build_grid_coords_on_178(hid_grid)   # (hid_grid^2, 2)
        self.hidden_coords = hidden_coords

        # Output mapping
        H, W = out_hw
        mask_out = make_circle_mask(H, W, radius=out_radius)           # (H,W) bool
        valid_lin = np.flatnonzero(mask_out.reshape(-1))               # circle pixels linear indices
        out100_all_178 = out100_to_178_coords(H, W)                    # (H*W, 2)
        output_coords_178 = out100_all_178[valid_lin]                  # (N_out_pix, 2)

        self.OUT_HW = (H, W)
        self.OUT_RADIUS = float(out_radius)
        self.valid_lin = valid_lin
        self.output_coords_178 = output_coords_178

        # Neighbor lists
        in_to_hidden = neighbor_lists(pos_cells, hidden_coords, r_in)
        hidden_to_out_circle = neighbor_lists(hidden_coords, output_coords_178, r_out)
        hidden_to_out_circle_rgb = [nb for _ in range(3) for nb in hidden_to_out_circle]

        # Build model + load weights
        model = LocalMLPCircle(self.N_cells, in_to_hidden, hidden_to_out_circle_rgb).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        print("Loaded model to", device)

        # Save state
        self.model = model
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.hid_grid = int(hid_grid)

    def prepare_input(self, mosaic_arr: np.ndarray) -> np.ndarray:
        """Return flat (N_cells,) vector from a 178x178 mosaic."""
        return mosaic_to_flat(mosaic_arr, self.idx_map, self.N_cells)

    @torch.no_grad()
    def predict(self, mosaic_arr: np.ndarray) -> np.ndarray:
        """
        Run a 178x178 mosaic through the model and return a (H,W,3) float32 image in [0,1],
        with the circle painted on black background.
        """
        flat = self.prepare_input(mosaic_arr)
        x = torch.from_numpy(flat[None, :]).to(self.device)  # (1, N_cells)
        y = self.model(x).squeeze(0).cpu().numpy()           # (3*N_out_pix,)

        H, W = self.OUT_HW
        pred = np.zeros((H, W, 3), dtype=np.float32)
        y_rgb = y.reshape(3, -1)
        for c in range(3):
            chan = np.zeros((H * W,), dtype=np.float32)
            chan[self.valid_lin] = y_rgb[c]
            pred[..., c] = chan.reshape(H, W)
        return np.clip(pred, 0.0, 1.0)
