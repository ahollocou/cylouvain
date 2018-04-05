#! /usr/bin/env python
#
# Copyright (C) 2018 Alexandre Hollocou <alexandre@hollocou.fr>
# License: 3-clause BSD

from setuptools import setup
from distutils.core import Extension
from setuptools.command.build_ext import build_ext as _build_ext

DISTNAME = 'cylouvain'
DESCRIPTION = 'Cython implementation of the classic Louvain algorithm for community detection in graphs'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
VERSION = '0.2.0'
AUTHOR = 'Alexandre Hollocou'
AUTHOR_EMAIL = 'alexandre@hollocou.fr'
URL = 'http://github.com/ahollocou/cylouvain'
LICENSE = 'new BSD'

IS_RELEASE = True

if not IS_RELEASE:
    try:
        from Cython.Build import cythonize
    except ImportError:
        USE_CYTHON = False
    else:
        USE_CYTHON = True
else:
    USE_CYTHON = False


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [
    Extension('cylouvain._louvain', ['cylouvain/_louvain' + ext],
              language="c++")]

if USE_CYTHON:
    extensions = cythonize(extensions)

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      license=LICENSE,
      packages=['cylouvain'],
      setup_requires=['numpy'],
      cmdclass={'build_ext': build_ext},
      install_requires=[
          'numpy', 'scipy', 'networkx',
      ],
      zip_safe=False,
      ext_modules=extensions)
