
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('msmtools', parent_package, top_path)

    config.add_subpackage('analysis')
    config.add_subpackage('analysis.dense')
    config.add_subpackage('analysis.sparse')
    config.add_subpackage('estimation')
    #config.add_subpackage('estimation')
    #config.add_subpackage('estimation')
    #config.add_subpackage('estimation')


    from Cython.Build import cythonize
    import sys
    config.ext_modules = cythonize(
        config.ext_modules,
        #compile_time_env={'SKLEARN_OPENMP_SUPPORTED': with_openmp},
        compiler_directives={'language_level': sys.version_info[0]})

    return config
