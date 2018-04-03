from setuptools import setup
from distutils.core import Extension
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True


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

setup(name='cylouvain',
      version='0.1.0',
      description='Cython implementation of the classic Louvain algorithm for community detection in graphs',
      url='http://github.com/ahollocou/cylouvain',
      author='Alexandre Hollocou',
      author_email='alexandre@hollocou.fr',
      license='new BSD',
      packages=['cylouvain'],
      setup_requires=['numpy'],
      cmdclass={'build_ext': build_ext},
      install_requires=[
          'numpy', 'scipy', 'networkx',
      ],
      zip_safe=False,
      ext_modules=extensions)
