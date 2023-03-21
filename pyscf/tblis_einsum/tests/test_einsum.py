import unittest
import numpy as np
from pyscf.tblis_einsum import tblis_einsum

def setUpModule():
    global bak
    tblis_einsum.EINSUM_MAX_SIZE, bak = 0, tblis_einsum.EINSUM_MAX_SIZE

def tearDownModule():
    global bak
    tblis_einsum.EINSUM_MAX_SIZE = bak

class KnownValues(unittest.TestCase):
    def test_d_d(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7))
        c0 = np.einsum('abcd,fdea->cebf', a, b)
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

        c1[:] = 0
        tblis_einsum.contract('abcd,fdea->cebf', a, b, out=c1)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

        c1 = np.empty_like(c0).transpose(0,3,2,1).copy('C')
        c1 = c1.transpose(0,3,2,1)
        tblis_einsum.contract('abcd,fdea->cebf', a, b, out=c1)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

        c1 = np.empty_like(c0).transpose(1,3,2,0).copy('C')
        c1 = c1.transpose(3,0,2,1)
        tblis_einsum.contract('abcd,fdea->cebf', a, b, out=c1)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_dslice(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7))
        c0 = np.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_dslice1(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7))
        c0 = np.einsum('abcd,fdea->cebf', a[:4].copy(), b[:,:,:,2:6])
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a[:4].copy(), b[:,:,:,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_d(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7))
        c0 = np.einsum('abcd,fdea->cebf', a[:,:,1:3,:], b)
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a[:,:,1:3,:], b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_dslice(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7))
        c0 = np.einsum('abcd,fdea->cebf', a[:,:,1:3], b[:,:,:2,:])
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a[:,:,1:3], b[:,:,:2,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_dslice1(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7))
        c0 = np.einsum('abcd,fdea->cebf', a[2:6,:,:1], b[:,:,1:3,2:6])
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a[2:6,:,:1], b[:,:,1:3,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_cslice(self):
        a = np.random.random((7,1,3,4))
        b = np.random.random((2,4,5,7)).astype(np.float32)
        c0 = np.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = tblis_einsum.contract('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_contraction4(self):
        a = np.random.random((5,2,3,2))
        b = np.random.random((2,4,7))
        c0 = np.einsum('...jkj,jlp->...jp', a, b)
        c1 = tblis_einsum.contract('...jkj,jlp->...jp', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction5(self):
        x = np.random.random((8,6))
        y = np.random.random((8,6,6))
        c0 = np.einsum("in,ijj->n", x, y)
        c1 = tblis_einsum.contract("in,ijj->n", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        x = np.random.random((6,6))
        y = np.random.random((8,8))
        c0 = np.einsum("ii,jj->", x, y)
        c1 = tblis_einsum.contract("ii,jj->", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        x = np.random.random((6))
        y = np.random.random((8))
        c0 = np.einsum("i,j->", x, y)
        c1 = tblis_einsum.contract("i,j->", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        c0 = np.einsum("i,i->", x, x)
        c1 = tblis_einsum.contract("i,i->", x, x)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        x = np.random.random((6,8))
        y = np.random.random((8,6))
        c0 = np.einsum("ij,ji->", x, y)
        c1 = tblis_einsum.contract("ij,ji->", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction6(self):
        d = np.random.random((2, 2, 10, 10))
        c = np.random.random((10, 10))
        res_np = np.einsum("iijk,kl->jl", d, c)
        res_tblis = tblis_einsum.contract("iijk,kl->jl", d, c)
        self.assertTrue (abs(res_np - res_tblis).max() < 1e-13)

        c = np.random.random((10, 10))
        res_np = np.einsum("ij,jk->k", c, c)
        res_tblis = tblis_einsum.contract("ij,jk->k", c, c)
        self.assertTrue (abs(res_np - res_tblis).max() < 1e-13)


if __name__ == '__main__':
    unittest.main()
