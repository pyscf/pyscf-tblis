# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

cmake_minimum_required (VERSION 3.5)
project (tblis_einsum)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  #TODO: test me set(CMAKE_MACOSX_RPATH On)
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif ()

include(ExternalProject)
ExternalProject_Add(libtblis
  GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
  GIT_TAG master
  PREFIX ${PROJECT_BINARY_DIR}/deps
  INSTALL_DIR ${PROJECT_SOURCE_DIR}
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>
          --disable-static CXX=${CMAKE_CXX_COMPILER}
)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR})

add_library(tblis_einsum SHARED as_einsum.cxx)
set_target_properties(tblis_einsum PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS "-std=c++11")
target_link_libraries(tblis_einsum tblis)
add_dependencies(tblis_einsum libtblis)
