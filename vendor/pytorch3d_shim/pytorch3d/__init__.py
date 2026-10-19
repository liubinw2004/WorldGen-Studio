"""pytorch3d shim - Windows compatibility replacement for the two pure-math
quaternion helpers WorldGen actually uses.

WorldGen only calls:
  * ``pytorch3d.transforms.matrix_to_quaternion``  (src/worldgen/utils/splat_utils.py:4, main path)
  * ``pytorch3d.transforms.quaternion_to_matrix``  (src/worldgen/pano_sharp.py:6, --use_sharp only)

This shim re-implements exactly those two functions with pure torch, matching
the official PyTorch3D semantics (wxyz convention, batch support, CUDA support).
It performs NO 3D operator kernels (knn / ball_query / chamfer / ...) because
WorldGen never uses them.

Registered onto sys.path via a ``.pth`` file (see scripts/install_pytorch3d_shim.bat);
the shim source itself is intentionally NOT copied into site-packages so it stays
auditable under <project>/vendor.
"""

__version__ = "0.7.9-shim"

from . import transforms  # noqa: F401

__all__ = ["transforms", "__version__"]
