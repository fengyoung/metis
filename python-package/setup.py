#! /ur/bin/env python

"""
setup.py file for metis predict
"""

from distutils.core import setup, Extension

utility_dir = "../src/utility/"
neural_network_dir = "../src/neural_network/"
prediction_dir = "../src/prediction/"

metis_predict_module = Extension('_metis_predict',
                                 include_dirs=[utility_dir, neural_network_dir],
                                 sources=['metis_predict_wrap.cxx',
                                          utility_dir + 'StringArray.cpp',
                                          utility_dir + 'Random.cpp',
                                          neural_network_dir + 'Matrix.cpp',
                                          neural_network_dir + 'TypeDefs.cpp',
                                          neural_network_dir + 'Activation.cpp',
                                          prediction_dir + 'Model.cpp',
                                          prediction_dir + 'Model_Perceptron.cpp',
                                          prediction_dir + 'Model_FM.cpp',
                                          prediction_dir + 'Model_MLP.cpp',
                                          prediction_dir + 'Model_FMSNN.cpp'])

setup(name='metis_predict',
      version='0.1',
      author='yujie6',
      description="""metis predict module""",
      ext_modules=[metis_predict_module],
      py_modules=["metis_predict"])
