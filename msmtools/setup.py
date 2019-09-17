
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('msmtools', parent_package, top_path)

    config.add_subpackage('analysis')
    # TODO: slash or dot??
    config.add_subpackage('analysis/dense')
    config.add_subpackage('analysis/sparse')

    config.add_subpackage('estimation')
    config.add_subpackage('flux')
    config.add_subpackage('generation')
    config.add_subpackage('util')

    import numpy as np
    import os
    config.add_include_dirs(np.get_include())
    config.add_include_dirs(os.path.abspath('util/include'))


    from Cython.Build import cythonize
    import sys
    config.ext_modules = cythonize(
        config.ext_modules,
        #compile_time_env={'SKLEARN_OPENMP_SUPPORTED': with_openmp},
        compiler_directives={'language_level': sys.version_info[0]})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
