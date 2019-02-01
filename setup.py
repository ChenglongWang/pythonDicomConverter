#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py
"""Setup script for pyDcmConverter."""
# Copyright (c) 2012-2017 Aditya Panchal
# This file is part of dicompyler, relased under a BSD license.
#    See the file license.txt included with this distribution, also
#    available at https://github.com/bastula/dicompyler/

from setuptools import setup, find_packages

requires = [
    #'matplotlib>=1.3.0',
    'numpy>=1.2.1',
    'pillow>=1.0',
    'dicompyler-core>=0.5.2',
    'pydicom>=0.9.9',
    'wxPython>=4.0.0b2',
    'utils-cw>=0.3']

setup(
    name="pyDcmConverter",
    version = "0.1",
    include_package_data = True,
    packages = find_packages(),
    package_data = {'pyDcmConverter': ['*.txt', 'resources/*.png', 'resources/*.xrc', 'resources/*.ico']},
    zip_safe = False,
    install_requires = requires,
    dependency_links = [
        'git+https://github.com/darcymason/pydicom.git#egg=pydicom-1.0.0',
        'git+https://github.com/dicompyler/dicompyler-core.git#egg=dicompyler-core-0.5.2',
        'git+https://github.com/ChenglongWang/py_utils_cw.git#egg=utils-cw-0.3'],
    entry_points={'console_scripts':['dcm-cvt = pyDcmConverter.dicomgui:main']},

    # metadata for upload to PyPI
    author = "Chenglong Wang",
    author_email = "cwang@mori.m.is.nagoya-u.ac.jp",
    description = "Simple DICOM file converter",
    license = "BSD License",
    keywords = "python dicom nifti rawdata"
)