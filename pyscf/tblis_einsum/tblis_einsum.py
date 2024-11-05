#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import os
import re
import ctypes
import numpy

libtblis = numpy.ctypeslib.load_library('libtblis_einsum', os.path.dirname(__file__))

libtblis.tensor_mult.restype = None
libtblis.tensor_mult.argtypes = (
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(), numpy.ctypeslib.ndpointer()
)

libtblis.tensor_add.restype = None
libtblis.tensor_add.argtypes = (
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(), numpy.ctypeslib.ndpointer()
)

libtblis.tensor_dot.restype = None
libtblis.tensor_dot.argtypes = (
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
    numpy.ctypeslib.ndpointer()
)

tblis_dtype = {
    numpy.dtype(numpy.float32)    : 0,
    numpy.dtype(numpy.double)     : 1,
    numpy.dtype(numpy.complex64)  : 2,
    numpy.dtype(numpy.complex128) : 3,
}

EINSUM_MAX_SIZE = 2000

def ctype_strides(*arrays):
    return [ (ctypes.c_size_t*arr.ndim)(*[x//arr.dtype.itemsize for x in arr.strides]) for arr in arrays ]

def ctype_shapes(*arrays):
    return [ (ctypes.c_size_t*arr.ndim)(*arr.shape) for arr in arrays ]

def check_tblis_shapes(a, a_inds, b, b_inds, subscripts=None, c_inds=None):
    a_shape_dic = dict(zip(a_inds, a.shape))
    b_shape_dic = dict(zip(b_inds, b.shape))
    if subscripts is None:
        subscripts = a_inds + ',' + b_inds
    if any(a_shape_dic[x] != b_shape_dic[x]
           for x in set(a_inds).intersection(b_inds)):
        raise ValueError('operands dimension error for "%s" : %s %s'
                         % (subscripts, a.shape, b.shape))

    if c_inds is not None:
        ab_shape_dic = a_shape_dic
        ab_shape_dic.update(b_shape_dic)
        c_shape = tuple([ab_shape_dic[x] for x in c_inds])
        return c_shape
    return None

_numpy_einsum = numpy.einsum
def contract(subscripts, *tensors, **kwargs):
    '''
    c = alpha * contract(a, b) + beta * c

    Args:
        tensors (list of ndarray) : Tensors for the operation.

    Kwargs:
        out (ndarray) : If provided, the calculation is done into this array.
        dtype (ndarray) : If provided, forces the calculation to use the data
            type specified.
        alpha (number) : Default is 1
        beta (number) :  Default is 0
    '''
    a = numpy.asarray(tensors[0])
    b = numpy.asarray(tensors[1])
    if not kwargs and (a.size < EINSUM_MAX_SIZE or b.size < EINSUM_MAX_SIZE):
        return _numpy_einsum(subscripts, a, b)

    c_dtype = kwargs.get('dtype', numpy.result_type(a, b))
    if (not (numpy.issubdtype(c_dtype, numpy.floating) or
             numpy.issubdtype(c_dtype, numpy.complexfloating))):
        return _numpy_einsum(subscripts, a, b)

    sub_idx = re.split(',|->', subscripts)
    indices  = ''.join(sub_idx)
    if '->' not in subscripts or any(indices.count(x) != 2 for x in set(indices)):
        return _numpy_einsum(subscripts, a, b)

    a_descr, b_descr, c_descr = sub_idx
    uniq_idxa = set(a_descr)
    uniq_idxb = set(b_descr)
    # Find the shared indices being summed over
    shared_idx = uniq_idxa.intersection(uniq_idxb)
    if ((not shared_idx) or  # Indices must overlap
        # repeated indices (e.g. 'iijk,kl->jl')
        len(a_descr) != len(uniq_idxa) or len(b_descr) != len(uniq_idxb)):
        return _numpy_einsum(subscripts, a, b)

    alpha = kwargs.get('alpha', 1)
    beta  = kwargs.get('beta', 0)
    c_dtype = numpy.result_type(c_dtype, alpha, beta)

    a = numpy.asarray(a, dtype=c_dtype)
    b = numpy.asarray(b, dtype=c_dtype)

    c_shape = check_tblis_shapes(a, a_descr, b, b_descr, subscripts=subscripts, c_inds=c_descr)

    out = kwargs.get('out', None)
    if out is None:
        order = kwargs.get('order', 'C')
        c = numpy.empty(c_shape, dtype=c_dtype, order=order)
    else:
        c = out
    return tensor_mult(a, a_descr, b, b_descr, c, c_descr, alpha=alpha, beta=beta, dtype=c_dtype)

def tensor_mult(a, a_inds, b, b_inds, c, c_inds, alpha=1, beta=0, dtype=None):
    ''' Wrapper for tblis_tensor_mult

    Performs the einsum operation
    c_{c_inds} = alpha * SUM[a_{a_inds} * b_{b_inds}] + beta * c_{c_inds}
    where the sum is over indices in a_inds and b_inds that are not in c_inds.
    '''

    if dtype is None:
        dtype = c.dtype.type
    assert dtype == c.dtype.type
    assert dtype == a.dtype.type
    assert dtype == b.dtype.type

    alpha = numpy.asarray(alpha, dtype=dtype)
    beta  = numpy.asarray(beta , dtype=dtype)

    assert len(a_inds) == a.ndim
    assert len(b_inds) == b.ndim
    assert len(c_inds) == c.ndim

    a_shape, b_shape, c_shape = ctype_shapes(a, b, c)
    a_strides, b_strides, c_strides = ctype_strides(a, b, c)
    assert c.shape == check_tblis_shapes(a, a_inds, b, b_inds, c_inds=c_inds)


    libtblis.tensor_mult(a, a.ndim, a_shape, a_strides, a_inds.encode('ascii'),
                       b, b.ndim, b_shape, b_strides, b_inds.encode('ascii'),
                       c, c.ndim, c_shape, c_strides, c_inds.encode('ascii'),
                       tblis_dtype[c.dtype], alpha, beta)
    return c


def tensor_add(a, a_inds, b, b_inds, alpha=1, beta=1):
    '''Wrapper for tblis_tensor_add
    '''
    assert a.dtype.type == b.dtype.type

    alpha = numpy.asarray(alpha, dtype=b.dtype)
    beta  = numpy.asarray(beta , dtype=b.dtype)

    assert len(a_inds) == a.ndim
    assert len(b_inds) == b.ndim

    a_shape, b_shape = ctype_shapes(a, b)
    a_strides, b_strides = ctype_strides(a, b)

    libtblis.tensor_add(a, a.ndim, a_shape, a_strides, a_inds.encode('ascii'),
                       b, b.ndim, b_shape, b_strides, b_inds.encode('ascii'),
                       tblis_dtype[b.dtype], alpha, beta)

def tensor_dot(a, a_inds, b, b_inds):
    '''Wrapper for tblis_tensor_dot
    '''

    assert a.dtype.type == b.dtype.type

    assert len(a_inds) == a.ndim
    assert len(b_inds) == b.ndim

    a_shape, b_shape = ctype_shapes(a, b)
    a_strides, b_strides = ctype_strides(a, b)

    result = numpy.zeros(1, dtype=a.dtype.type)

    libtblis.tensor_dot(a, a.ndim, a_shape, a_strides, a_inds.encode('ascii'),
                       b, b.ndim, b_shape, b_strides, b_inds.encode('ascii'),
                       tblis_dtype[b.dtype], result)

    return result[0]
