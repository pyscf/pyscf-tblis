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
from setuptools.command.build_ext import build_ext


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

class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == 'libtblis_einsum'
        self.build_cmake(extension)

    def build_cmake(self, extension):
        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'pyscf', 'tblis_einsum'))
        cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}']
        if sys.platform.startswith('linux'):
            # When compiling tblis, the configure command needs to be
            # export ORIGIN='$ORIGIN'
            # ../configure LDFLAGS='-Wl,-rpath -Wl,$$ORIGIN -Wl,-z,origin')
            # See https://stackoverflow.com/questions/62034115/difficulty-adding-origin-to-ldflags
            os.environ['ORIGIN'] = '$ORIGIN'
            os.environ['LDFLAGS'] = '-Wl,-rpath -Wl,$$ORIGIN -Wl,-z,origin'
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        cmd = ['cmake', '--build', self.build_temp, '-j4']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

    # To remove the infix string like cpython-37m-x86_64-linux-gnu.so
    # Python ABI updates since 3.5
    # https://www.python.org/dev/peps/pep-3149/
    def get_ext_filename(self, ext_name):
        ext_path = ext_name.split('.')
        filename = build_ext.get_ext_filename(self, ext_name)
        name, ext_suffix = os.path.splitext(filename)
        return os.path.join(*ext_path) + ext_suffix

# Here to change the order of sub_commands to ['build_py', ..., 'build_ext']
# C extensions by build_ext are installed in source directory.
# build_py then copy all .so files into "build_ext.build_lib" directory.
# We have to ensure build_ext being executed earlier than build_py.
# A temporary workaround is to modifying the order of sub_commands in build class
from distutils.command.build import build
build.sub_commands = ([c for c in build.sub_commands if c[0] == 'build_ext'] +
                      [c for c in build.sub_commands if c[0] != 'build_ext'])

settings = {
    'name': metadata.get('NAME', None),
    'version': VERSION,
    'description': metadata.get('DESCRIPTION', None),
    'author': metadata.get('AUTHOR', None),
    'author_email': metadata.get('AUTHOR_EMAIL', None),
    'install_requires': metadata.get('DEPENDENCIES', []),
    'ext_modules': [Extension('libtblis_einsum', [])],
    'cmdclass': {'build_ext': CMakeBuildExt},
}
setup(
    include_package_data=True,
    packages=modules,
    **settings
)
