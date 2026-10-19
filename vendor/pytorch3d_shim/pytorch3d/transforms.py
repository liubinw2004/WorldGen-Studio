"""Pure-torch quaternion transforms, API-compatible with ``pytorch3d.transforms``.

Only the two functions used by WorldGen are provided:
    * :func:`matrix_to_quaternion`
    * :func:`quaternion_to_matrix`

Both follow the official PyTorch3D convention: quaternions are ``(w, x, y, z)``,
support arbitrary leading batch dimensions ``(..., 3, 3)`` / ``(..., 4)`` and work
on CUDA tensors.

The implementations below are faithful ports of the upstream PyTorch3D algorithms
(Shepperd's method with the "best candidate" selection for numerical stability, and
the standard closed-form rotation-matrix reconstruction).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["matrix_to_quaternion", "quaternion_to_matrix", "standardize_quaternion"]


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(max(0, x))`` element-wise (clamp negatives to 0)."""
    return torch.sqrt(torch.clamp(x, min=0.0))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Return a quaternion given in wxyz form, with positive real part (w >= 0).

    Args:
        quaternions: tensor of shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Tensor of the same shape with the real part made non-negative.
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: tensor of shape ``(..., 4)`` holding ``(w, x, y, z)``.
            Quaternions may be non-unit; they are normalised internally.

    Returns:
        Rotation matrices as tensor of shape ``(..., 3, 3)``.
    """
    if not torch.is_tensor(quaternions):
        quaternions = torch.as_tensor(quaternions, dtype=torch.float32)
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {tuple(quaternions.shape)}.")

    r, i, j, k = torch.unbind(quaternions, -1)
    # two_s = 2.0 / (q . q); guard against a zero quaternion.
    two_s = 2.0 / torch.clamp((quaternions * quaternions).sum(-1), min=1e-30)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: rotation matrices as tensor of shape ``(..., 3, 3)``.

    Returns:
        Quaternions as tensor of shape ``(..., 4)`` in ``(w, x, y, z)`` order,
        with non-negative real part.
    """
    if not torch.is_tensor(matrix):
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {tuple(matrix.shape)}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # Each of the four candidates below is the quaternion scaled by r, i, j, k
    # respectively; the numerically-largest component (q_abs.argmax) is the best.
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # Floor the denominators at 0.1 to avoid division by (near) zero.
    # q_abs[..., None] has a trailing singleton dim, so the max over dim=-1 is
    # simply the clamped component itself; we clamp directly (torch.Tensor.max
    # does not accept a (other, dim, keepdim) signature).
    denom = 2.0 * torch.clamp(q_abs, min=0.1)[..., None]
    quat_candidates = quat_by_rijk / denom

    # Select the candidate with the largest q_abs component for each element.
    idx = q_abs.argmax(dim=-1)
    one_hot = F.one_hot(idx, num_classes=4) > 0.5
    out = quat_candidates[one_hot, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out)
