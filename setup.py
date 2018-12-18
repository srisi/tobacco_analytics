from setuptools import setup
from Cython.Build import cythonize
import numpy as np

def run():
    setup(name='tobacco',
          version = '0.16',
          author="Stephan Risi",
          author_email="risi@stanford.edu",
          packages=['tobacco',
                    #'tobacco.aws',
                    #'tobacco.correlations',
                    'tobacco.data',
                    'tobacco.frequencies',
                    'tobacco.frequencies_preprocessing',
                    'tobacco.full_db_preprocessing',
                    #'tobacco.litigation',
                    #'tobacco.networks',
                    #'tobacco.results_storage',
                    #'tobacco.stats',
                    #'tobacco.text_passages',
                    'tobacco.utilities'
                    ],
          include_package_data=True,
          package_data= {'': ['data.*', 'utilities.*'],},
          ext_modules=cythonize([#'tobacco/correlations/*.pyx',
                                 'tobacco/full_db_preprocessing/*.pyx',
                                 'tobacco/frequencies_preprocessing/*.pyx',
                                 #'tobacco/stats/*.pyx',
                                 #'tobacco/text_passages/*.pyx',
                                 'tobacco/utilities/*.pyx'
          ]),
          include_dirs=[np.get_include(), '.'],
          zip_safe=False,
          )

'''
Requirements

pip install pyenchant marisa_trie mysqlclient boto fabric

'''


if __name__ =="__main__":
    run()