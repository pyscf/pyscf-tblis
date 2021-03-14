#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NAME = 'pyscf-tblis'
AUTHOR = 'Pyscf Developer'
AUTHOR_EMAIL = None
DESCRIPTION  = 'A wrapper of TBLIS library'
DEPENDENCIES = ['numpy']

#######################################################################
# Unless not working, nothing below needs to be changed.
metadata = globals()
import os
import sys
from setuptools import setup, find_namespace_packages, Extension

topdir = os.path.abspath(os.path.join(__file__, '..'))
modules = find_namespace_packages(include=['pyscf.*'])
def guess_version():
    for module in modules:
        module_path = os.path.join(topdir, *module.split('.'))
        for version_file in ['__init__.py', '_version.py']:
            version_file = os.path.join(module_path, version_file)
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    for line in f.readlines():
                        if line.startswith('__version__'):
                            delim = '"' if '"' in line else "'"
                            return line.split(delim)[1]
    raise ValueError("Version string not found")
if not metadata.get('VERSION', None):
    VERSION = guess_version()

settings = {
    'name': metadata.get('NAME', None),
    'version': VERSION,
    'description': metadata.get('DESCRIPTION', None),
    'author': metadata.get('AUTHOR', None),
    'author_email': metadata.get('AUTHOR_EMAIL', None),
    'install_requires': metadata.get('DEPENDENCIES', []),
}
setup(
    include_package_data=True,
    packages=modules,
    **settings
)
