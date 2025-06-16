from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    include_paths,
)
from platform import system

NAME = 'bioeq'
VERSION = '0.1.0'
LICENSE = 'CC BY-NC 4.0'
FILES = [
    'bioeq/src/mol.cpp',
    'bioeq/src/bind.cpp',
]

EXT_NAME = 'bioeq.src._cpp'
ext_modules = []
extra_compile_args = [
    "-O3",
    "-std=c++20",
]
if system() == 'Darwin':
    extra_compile_args += [
        "-stdlib=libc++",
        "-fopenmp=libomp"
    ]
CPP_EXT = CppExtension(
    EXT_NAME,
    FILES,
    include_dirs=include_paths(),
    extra_compile_args=extra_compile_args,
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
