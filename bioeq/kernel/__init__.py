try:
    from . import _C
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
