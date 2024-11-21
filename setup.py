from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    include_paths,
)

NAME = 'bioeq'
VERSION = '0.1.0'
LICENSE = 'CC BY-NC 4.0'
FILES = [
    'bioeq/_cpp/mol.cpp',
    'bioeq/_cpp/bind.cpp',
]

EXT_NAME = 'bioeq._cpp._c'
ext_modules = []
CPP_EXT = CppExtension(
    EXT_NAME,
    FILES,
    include_dirs=include_paths(),
    extra_compile_args=[
        "-O3",
        "-stdlib=libc++",
        "-std=c++20",
    ]
)
ext_modules.append(CPP_EXT)

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
