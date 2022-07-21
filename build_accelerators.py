# Portable way to build accelerators library
# 
# Alternatively you could build manually. For example:
# gcc -Wall -Wextra -O3 -march=native -shared extract_helper.c -o extract_helper.dll

from distutils.ccompiler import new_compiler
import sys
from pathlib import Path
import shutil
import tempfile


def build_to(target_dir):
    cc = new_compiler()
    if sys.platform == 'win32':
        extra_args = ['/O2']  # Assume MSVC on Windows
    else:
        extra_args = ['-O3']  # Assume gcc-compatible elsewhere
    
    sources = ['extract_helper.c']
    target = cc.shared_object_filename('extract_helper')

    with tempfile.TemporaryDirectory() as build_dir:        
        objects = cc.compile(sources, output_dir=build_dir, extra_postargs=extra_args)

        cc.link_shared_object(objects, target, output_dir=build_dir)

        shutil.copyfile(Path(build_dir) / target, Path(target_dir) / target)


if __name__ == '__main__':
    build_to(Path(__file__).parent)
