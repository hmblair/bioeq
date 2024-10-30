from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    include_paths,
)

NAME = 'bioeq'
VERSION = '0.1.0'
LICENSE = 'CC BY-NC 4.0'
FILES = [
    'bioeq/kernel/src/bind.cpp',
    'bioeq/kernel/src/kernel.cu',
]

KERNEL_NAME = 'bioeq.kernel._C'
# Conditionally add the CUDA extension if CUDA is available
ext_modules = []
if torch.cuda.is_available():
    CUDA_EXT = CUDAExtension(
        KERNEL_NAME,
        FILES,
        include_dirs=include_paths(True),
        extra_compile_args={
            'nvcc': ['-Xptxas="-v"'],
            'cxx': ['-O3'],
        }
    )
    ext_modules.append(CUDA_EXT)
else:
    print("CUDA is not available. Skipping CUDA extension build.")

AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/bioeq'

setup(
    name=NAME,
    version=VERSION,
    ext_modules=ext_modules,
    packages=find_packages(),
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=[
        'torch',
    ],
)
