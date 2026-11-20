"""Load an existing gsplat CUDA cache without forcing a JIT rebuild."""

import os
import sys
import types

from torch.utils.cpp_extension import (
    _get_build_directory,
    _import_module_from_library,
)


def preload_cached_gsplat():
    """Register a previously built gsplat extension when one is available.

    gsplat 1.5.3 calls PyTorch's JIT version checker even when a working
    extension already exists. In this environment that can trigger a rebuild
    against incompatible camera headers. Directly importing the same cached
    binary preserves the extension used by the completed baseline runs.
    """
    module_name = 'gsplat.cuda._backend'
    if module_name in sys.modules:
        return True

    build_directory = _get_build_directory('gsplat_cuda', verbose=False)
    library_path = os.path.join(build_directory, 'gsplat_cuda.so')
    if not os.path.isfile(library_path):
        return False

    extension = _import_module_from_library(
        'gsplat_cuda', build_directory, is_python_module=True)
    backend = types.ModuleType(module_name)
    backend._C = extension
    backend.__file__ = library_path
    sys.modules[module_name] = backend
    return True
