import sys
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from worldlayers import __version__


def shared_object_ext():
    return {
        'win32': '.dll',
        'cygwin': '.dll',
        'darwin': '.dylib',
    }.get(sys.platform, '.so')

class PlainLibrary(Extension):
    pass

class build_ext_mod(build_ext):
    def get_ext_filename(self, ext_name):
        if isinstance(self.ext_map[ext_name], PlainLibrary):
            return ext_name + shared_object_ext()
        return super().get_ext_filename(ext_name)
    
    def get_export_symbols(self, ext):
        if isinstance(ext, PlainLibrary):
            return []
        return super().get_export_symbols(ext)


with open('requirements.txt') as file:
    REQUIREMENTS = [
        line for line in map(str.strip, file)
        if line and not line.startswith('-e')
    ]

with open('README.md', encoding='utf-8') as file:
    README = file.read()


setup(
    name='worldlayers',
    version=__version__,
    description='Create block distribution graphs and statistics of Minecraft worlds',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/Metaray/WorldLayers',
    license='MIT',
    packages=['worldlayers'],
    python_requires='>=3.9',
    install_requires=REQUIREMENTS,
    ext_modules=[
        PlainLibrary('worldlayers.extract_helper', ['worldlayers/extract_helper.c'])
    ],
    cmdclass={'build_ext': build_ext_mod},
    entry_points={
        'console_scripts': [
            'worldlayers = worldlayers.cli:main',
        ],
    },
)
