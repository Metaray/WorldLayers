# Portable way to build accelerators shared library
# 
# Alternatively you could build manually. For example:
# gcc -Wall -Wextra -O3 -march=native -shared extract_helper.c -o extract_helper.dll

try:
    from setuptools._distutils.ccompiler import new_compiler
except ImportError:
    from distutils.ccompiler import new_compiler

import sys
from pathlib import Path
import shutil
import tempfile


def shared_object_ext():
    return {
        'win32': '.dll',
        'cygwin': '.dll',
        'darwin': '.dylib',
    }.get(sys.platform, '.so')


def build_so(name, sources, target_dir, сс_name=None):
    cc = new_compiler(compiler=сс_name)
    if cc.compiler_type == 'msvc':
        extra_args = ['/O2']
    else:
        extra_args = ['-O3']  # Assume gcc-compatible otherwise
    
    target = name + shared_object_ext()

    with tempfile.TemporaryDirectory() as build_dir:
        objects = cc.compile(sources, output_dir=build_dir, extra_postargs=extra_args)

        cc.link_shared_object(objects, target, output_dir=build_dir)

        shutil.copyfile(Path(build_dir) / target, Path(target_dir) / target)


cc_name = None
if len(sys.argv) > 1:
    cc_name = sys.argv[1]

target_dir = Path(__file__).parent

build_so('extract_helper', ['extract_helper.c'], target_dir, cc_name)
