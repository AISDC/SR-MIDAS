"""GPU-side per-frame preprocessing and connected-component labeling.

Mirrors the per-frame CPU preprocessing in `sr_process.py` (image
transforms, dark subtraction, flood division, beam-current scaling,
bad-pixel zeroing, mask, ring-thresholding) but keeps the frame on the
GPU as a torch tensor. Provides `gpu_ccl()` for batched-mask connected
component labeling via iterative 3x3 min-pooling.

These functions are used by the experimental all-GPU pipeline driver in
`claude_docs/benchmarks/`. The existing `sr_process.py` workflow is not
affected.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def upload_correction_frames(dark_np, flood_np, mask_np, ring_nr_map_np,
                             ImTransOpt, torch_devs):
    """One-time upload of detector correction arrays to GPU.

    These arrays are static for the whole scan; uploading once at
    pipeline startup avoids re-transferring them every frame.
    `ImTransOpt` is recorded for documentation but not applied here —
    the caller is expected to have already applied it to dark/flood/
    mask/ring_nr_map upstream (mirroring `sr_process.py:198-207`).

    Args:
        dark_np:        (H, W) numpy, dark frame to subtract.
        flood_np:       (H, W) numpy, flood-field divisor.
        mask_np:        (H, W) numpy, pixels where mask>0 get zeroed.
        ring_nr_map_np: (H, W) numpy int, per-pixel ring index (-1 for
            none, 0..N-1 for active rings).
        ImTransOpt:     list[int]; currently unused (transforms must be
            pre-applied), kept in the signature for future hook points.
        torch_devs:     torch.device, the target CUDA device.

    Returns:
        dict with keys `"dark"`, `"flood"`, `"mask"`, `"ring_nr_map"` —
        each a torch tensor on `torch_devs` (float32 for the first three,
        int32 for `ring_nr_map`).
    """
    return {
        "dark": torch.from_numpy(dark_np.astype(np.float32)).to(torch_devs),
        "flood": torch.from_numpy(flood_np.astype(np.float32)).to(torch_devs),
        "mask": torch.from_numpy(mask_np.astype(np.float32)).to(torch_devs),
        "ring_nr_map": torch.from_numpy(ring_nr_map_np.astype(np.int32)).to(torch_devs),
    }


def preprocess_frame_gpu(frame_np, gpu_corr, sr_params):
    """Build `frame_goodCoords` on GPU for a single raw detector frame.

    Reproduces the per-frame CPU preprocessing chain in
    `sr_process.py:398-424` end-to-end on device:
      1. H2D the raw frame (uint16 typical) as float32.
      2. Apply `ImTransOpt` transforms in order (1=flipY, 2=flipZ, 3=transpose).
      3. `(frame - dark) / flood * bc`.
      4. Zero pixels equal to `BadPxIntensity` (if nonzero in config).
      5. Zero pixels where `mask > 0`.
      6. For each ring index `i`, keep `arr` pixels where
         `ring_nr_map == i` AND `arr >= ringsThresh[i]`; zero elsewhere.

    Args:
        frame_np: (H, W) numpy raw detector frame (any numeric dtype).
        gpu_corr: dict returned by `upload_correction_frames`.
        sr_params: dict; needs `ImTransOpt`, `bc`, `BadPxIntensity`,
            `ringsThresh`.

    Returns:
        (H, W) torch float32 on the same device as `gpu_corr["dark"]`,
        with non-good pixels (off-ring or below threshold) set to 0.
    """
    device = gpu_corr["dark"].device
    arr = torch.from_numpy(frame_np.astype(np.float32)).to(device)
    for opt in sr_params["ImTransOpt"]:
        if opt == 1:
            arr = torch.flip(arr, dims=[1])
        elif opt == 2:
            arr = torch.flip(arr, dims=[0])
        elif opt == 3:
            arr = arr.t().contiguous()

    arr = (arr - gpu_corr["dark"]) / gpu_corr["flood"]
    arr = arr * float(sr_params["bc"])

    if sr_params["BadPxIntensity"] != 0:
        arr = torch.where(arr == float(sr_params["BadPxIntensity"]),
                          torch.zeros_like(arr), arr)
    arr = torch.where(gpu_corr["mask"] > 0, torch.zeros_like(arr), arr)

    ring_nr_map = gpu_corr["ring_nr_map"]
    good = torch.zeros_like(arr)
    for ring_i, thresh in enumerate(sr_params["ringsThresh"]):
        m = (ring_nr_map == ring_i) & (arr >= float(thresh))
        good = torch.where(m, arr, good)
    return good


def gpu_ccl(binary_mask_t, max_iter=512):
    """Connected-component labeling on GPU via iterative 3x3 min-propagation.

    Equivalent to `scipy.ndimage.label(arr, structure=np.ones((3,3)))`
    (8-connectivity): produces the same partition of foreground pixels
    into components (the integer label values differ; the equivalence
    classes are identical — verified on real testSR frames).

    Convergence takes O(longest-component-diameter) iterations of a
    single 3x3 max-pool each — fine for diffraction spots (max diameter
    ~20-30 px, observed 9-16 iterations on testSR). `max_iter` caps the
    loop as a safety against pathological inputs (e.g. long thin
    streaks spanning the detector) where convergence would be O(image
    diameter).

    Args:
        binary_mask_t: (H, W) torch bool (or castable) on GPU; True
            marks foreground pixels.
        max_iter:      int, hard cap on iterations (default 512).

    Returns:
        labels: (H, W) torch int32; background = 0, components numbered
            1..K consecutively (matching scipy's convention).
        K:      int, number of distinct connected components found.
    """
    binary_mask_t = binary_mask_t.bool()
    H, W = binary_mask_t.shape
    device = binary_mask_t.device

    INF = float(H * W + 2)
    flat_idx = (torch.arange(H * W, device=device, dtype=torch.float32)
                .reshape(H, W) + 1.0)
    labels = torch.where(binary_mask_t, flat_idx,
                         torch.full((H, W), INF, device=device,
                                    dtype=torch.float32))
    for _ in range(max_iter):
        prev = labels
        pooled = -F.max_pool2d((-labels).unsqueeze(0).unsqueeze(0),
                               kernel_size=3, stride=1, padding=1)
        new_labels = pooled.squeeze(0).squeeze(0)
        labels = torch.where(binary_mask_t,
                             torch.minimum(labels, new_labels),
                             torch.full_like(labels, INF))
        if torch.equal(labels, prev):
            break

    fg_vals = labels[binary_mask_t]
    if fg_vals.numel() == 0:
        return torch.zeros((H, W), device=device, dtype=torch.int32), 0
    uniq, inverse = torch.unique(fg_vals, return_inverse=True)
    out = torch.zeros((H, W), device=device, dtype=torch.int32)
    out[binary_mask_t] = (inverse + 1).to(torch.int32)
    return out, int(uniq.numel())


def extract_patches_gpu(good_t, labels_t, sr_config):
    """GPU equivalent of `patches_from_detector_frame`.

    Pure-GPU patch extraction: per-label argmax via two scatter_reduce
    passes (max value, then min-position tie-break), then a single batched
    gather to slice each (patch_size, patch_size) window centered on its
    max pixel. Pixels outside the spot's label are zeroed (matches the
    CPU code's `spot = where(labels==i+1, frame, 0)` step).

    Args:
        good_t:  (H, W) torch tensor, frame_goodCoords on device
        labels_t: (H, W) torch int32 tensor, 0=background, 1..N components
        sr_config: dict with `spot_find_args.patch_size` and `minPxCount`

    Returns:
        patches_t:    (B, 1, P, P) float32 on device
        Y00_t, Z00_t: (B,) int64 on device — patch-origin pixel coords
        nr_pixels_t:  (B,) int64 on device — pixel count per spot
    """
    H, W = good_t.shape
    P = int(sr_config["spot_find_args"]["patch_size"])
    P2 = P // 2
    device = good_t.device

    labels_flat = labels_t.flatten().long()
    good_flat = good_t.flatten().to(torch.float32)

    n_labels = int(labels_t.max().item()) + 1   # +1 for the background bin
    if n_labels <= 1:
        return (torch.empty(0, 1, P, P, device=device, dtype=torch.float32),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))

    pixel_count = torch.bincount(labels_flat, minlength=n_labels)
    pixel_count[0] = 0  # background

    max_per_label = torch.full((n_labels,), -float("inf"), device=device, dtype=torch.float32)
    max_per_label.scatter_reduce_(0, labels_flat, good_flat, reduce="amax", include_self=True)

    is_max = (good_flat == max_per_label[labels_flat]) & (labels_flat > 0)
    positions = torch.arange(H * W, device=device, dtype=torch.long)
    pos_or_inf = torch.where(is_max, positions, torch.full_like(positions, H * W))
    argmax_per_label = torch.full((n_labels,), H * W, device=device, dtype=torch.long)
    argmax_per_label.scatter_reduce_(0, labels_flat, pos_or_inf, reduce="amin", include_self=True)

    Zpx_max = argmax_per_label // W
    Ypx_max = argmax_per_label %  W
    Z00 = Zpx_max - P2
    Y00 = Ypx_max - P2

    valid = (pixel_count >= int(sr_config["minPxCount"])) \
          & (Z00 >= 0) & (Z00 < H - P) \
          & (Y00 >= 0) & (Y00 < W - P)
    valid[0] = False
    spot_ids = torch.nonzero(valid, as_tuple=False).flatten()
    B = int(spot_ids.shape[0])
    if B == 0:
        return (torch.empty(0, 1, P, P, device=device, dtype=torch.float32),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))

    Zpx_v = Zpx_max[spot_ids]
    Ypx_v = Ypx_max[spot_ids]
    Y00_v = Y00[spot_ids]
    Z00_v = Z00[spot_ids]
    npx_v = pixel_count[spot_ids]

    d = torch.arange(P, device=device, dtype=torch.long) - P2
    dz = d.view(1, P, 1)
    dy = d.view(1, 1, P)
    z = (Zpx_v.view(B, 1, 1) + dz).expand(B, P, P)
    y = (Ypx_v.view(B, 1, 1) + dy).expand(B, P, P)
    flat_idx = z * W + y  # within bounds because the validity filter clipped both axes

    frame_vals = good_flat[flat_idx]
    label_at = labels_flat[flat_idx]
    spot_id_b = spot_ids.view(B, 1, 1)
    mask = (label_at == spot_id_b)
    patches = torch.where(mask, frame_vals, torch.zeros_like(frame_vals))

    return patches.unsqueeze(1), Y00_v, Z00_v, npx_v
