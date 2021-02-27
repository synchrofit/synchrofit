
#! /usr/bin/env python
"""
Setup for synchrofit
"""

import os
from setuptools import find_packages
from setuptools import setup

def read(fname):
    """Read a file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

reqs = ['pandas>=1.2.2',
        'numba>=0.52.0',
        'scipy>=1.6.1',
        'matplotlib>=3.3.4',
        'argparse>=1.1']

setup(
    name='synchrofit',
    version='1.0.0',
    description='A package for modelling radio spectra',
    author='Ross J Turner, Ben Quici',
    author_email='turner.rj@icloud.com, benjamin.quici@postgrad.curtin.edu.au',
    url='https://github.com/synchrofit/',
    license='MIT',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=reqs,
    packages=['SynchrofitTools'],
    scripts=["scripts/synchrofit"],
    python_requires='>=2.7',
)