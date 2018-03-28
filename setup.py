from distutils.core import setup
from distutils.core import Extension

try:
    import numpy

    try:
        from Cython.Build import cythonize
    except ImportError:
        USE_CYTHON = False
    else:
        USE_CYTHON = True

    ext = '.pyx' if USE_CYTHON else '.cpp'

    extensions = [
        Extension('cylouvain._louvain', ['cylouvain/_louvain' + ext],
                  include_dirs=[numpy.get_include()],
                  language="c++")]

    if USE_CYTHON:
        extensions = cythonize(extensions)

    setup(name='cylouvain',
          version='0.0.1',
          description='Cython implementation of the classic Louvain algorithm for community detection in graphs',
          url='http://github.com/ahollocou/cylouvain',
          author='Alexandre Hollocou',
          author_email='alexandre@hollocou.fr',
          license='new BSD',
          packages=['cylouvain'],
          install_requires=[
              'numpy', 'scipy', 'networkx',
          ],
          zip_safe=False,
          ext_modules=extensions)

except ImportError:
    raise ImportError(
        "The installation requires NumPy. In order to install NumPy, run the 'pip install numpy' command.")
