"""P2P accessibility registry for multi-GPU DLPack operations.

Caches cudaDeviceCanAccessPeer results per GPU pair to avoid
repeated CUDA runtime API calls.
"""

import ctypes
import glob
import logging
import os
import sys
import torch

logger = logging.getLogger("MultiGPU")

_libcudart = None
_libcudart_failed = False


def _candidate_cudart_paths():
    """Yield platform-specific paths/names for the CUDA runtime library."""
    if sys.platform == "win32":
        names = ["cudart64_*.dll"]
    else:
        names = ["libcudart.so*", "libcuda.so*"]
    search_dirs = [os.getcwd()]
    try:
        sp_dir = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
        nvidia_rt = os.path.join(sp_dir, "nvidia", "cuda_runtime", "bin")
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        search_dirs += [nvidia_rt, torch_lib]
    except Exception:
        pass
    if sys.platform == "win32":
        for base in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                     r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"):
            search_dirs += glob.glob(os.path.join(base, "v*", "bin"))
    for d in search_dirs:
        for pat in names:
            for p in sorted(glob.glob(os.path.join(d, pat)), reverse=True):
                yield p


def _get_libcudart():
    """Load the CUDA runtime library once and cache the handle.

    Returns None if no loadable cudart can be found (e.g. headless or
    unusual installs); callers must treat that as 'P2P unknown' rather
    than crashing, so cross-device work falls back to CPU staging.
    """
    global _libcudart, _libcudart_failed
    if _libcudart is not None:
        return _libcudart
    if _libcudart_failed:
        return None
    try:
        # Fast path: plain name (works on Linux; Windows usually needs a path)
        _libcudart = ctypes.CDLL("cudart64_13.dll" if sys.platform == "win32" else "libcudart.so")
        return _libcudart
    except OSError:
        pass
    for p in _candidate_cudart_paths():
        try:
            _libcudart = ctypes.CDLL(p)
            logger.info(f"[MultiGPU P2P] Loaded CUDA runtime from {p}")
            return _libcudart
        except OSError:
            continue
    logger.warning(
        "[MultiGPU P2P] Could not locate a loadable CUDA runtime library; "
        "assuming no P2P (cross-device DLPack tensors will be CPU-staged)"
    )
    _libcudart_failed = True
    return None


class MultiGPUP2PRegistry:
    """Cached registry for CUDA peer-to-peer accessibility between GPU pairs.

    Uses the CUDA runtime cudaDeviceCanAccessPeer API directly via ctypes
    because torch.cuda.can_access_peer does not exist in PyTorch 2.10.0+.
    Results are cached per (src, dst) pair for the lifetime of the registry.
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _raw_can_access_peer(device_a: int, device_b: int) -> bool:
        """Call cudaDeviceCanAccessPeer via ctypes. Returns True if P2P is available."""
        lib = _get_libcudart()
        if lib is None:
            return False
        can_access = ctypes.c_int(0)
        result = lib.cudaDeviceCanAccessPeer(ctypes.byref(can_access), device_a, device_b)
        if result != 0:
            logger.warning(
                f"[MultiGPU P2P] cudaDeviceCanAccessPeer({device_a}, {device_b}) "
                f"returned error code {result}, assuming no P2P"
            )
            return False
        return bool(can_access.value)

    def can_access_peer(self, src_device: int, dst_device: int) -> bool:
        """Check if src_device can access dst_device memory via P2P.

        Results are cached per (src, dst) pair.
        """
        if src_device == dst_device:
            return True

        key = (src_device, dst_device)
        if key not in self._cache:
            if not torch.cuda.is_available():
                self._cache[key] = False
            else:
                result = self._raw_can_access_peer(src_device, dst_device)
                self._cache[key] = result
                logger.info(
                    f"[MultiGPU P2P] can_access_peer({src_device}, {dst_device}) = {result}"
                )
        return self._cache[key]

    def clear_cache(self):
        """Clear the P2P cache (useful for testing)."""
        self._cache.clear()


# Module-level singleton
p2p_registry = MultiGPUP2PRegistry()
