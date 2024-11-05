#include <stddef.h>
#include <stdlib.h>
#include <tblis/tblis.h>
using namespace tblis;

static void _as_tensor(tblis_tensor *t, void *data, int dtype, int ndim,
                       ptrdiff_t *shape, ptrdiff_t *strides, void *scalar)
{
    switch (dtype) {
        case TYPE_SINGLE:
            if (scalar == NULL) {
                tblis_init_tensor_s(t, ndim, shape, static_cast<float *>(data), strides);
            } else {
                tblis_init_tensor_scaled_s(t, *(static_cast<float *>(scalar)), ndim, shape,
                                           static_cast<float *>(data), strides);
            }
            break;
        case TYPE_DOUBLE:
            if (scalar == NULL) {
                tblis_init_tensor_d(t, ndim, shape, static_cast<double *>(data), strides);
            } else {
                tblis_init_tensor_scaled_d(t, *(static_cast<double *>(scalar)), ndim, shape,
                                           static_cast<double *>(data), strides);
            }
            break;
        case TYPE_SCOMPLEX:
            if (scalar == NULL) {
                tblis_init_tensor_c(t, ndim, shape, static_cast<scomplex *>(data), strides);
            } else {
                tblis_init_tensor_scaled_c(t, *(static_cast<scomplex *>(scalar)), ndim, shape,
                                           static_cast<scomplex *>(data), strides);
            }
            break;
        case TYPE_DCOMPLEX:
            if (scalar == NULL) {
                tblis_init_tensor_z(t, ndim, shape, static_cast<dcomplex *>(data), strides);
            } else {
                tblis_init_tensor_scaled_z(t, *(static_cast<dcomplex *>(scalar)), ndim, shape,
                                           static_cast<dcomplex *>(data), strides);
            }
            break;
    }
}

extern "C" {
void tensor_mult(void *data_A, int ndim_A, ptrdiff_t *shape_A, ptrdiff_t *strides_A, char *descr_A,
               void *data_B, int ndim_B, ptrdiff_t *shape_B, ptrdiff_t *strides_B, char *descr_B,
               void *data_C, int ndim_C, ptrdiff_t *shape_C, ptrdiff_t *strides_C, char *descr_C,
               int dtype, void *alpha, void *beta)
{
    tblis_tensor A, B, C;
    _as_tensor(&A, data_A, dtype, ndim_A, shape_A, strides_A, alpha);
    _as_tensor(&B, data_B, dtype, ndim_B, shape_B, strides_B, NULL);
    _as_tensor(&C, data_C, dtype, ndim_C, shape_C, strides_C, beta);

    tblis_tensor_mult(NULL, NULL, &A, descr_A, &B, descr_B, &C, descr_C);
}


void tensor_add(void *data_A, int ndim_A, ptrdiff_t *shape_A, ptrdiff_t *strides_A, char *descr_A,
               void *data_B, int ndim_B, ptrdiff_t *shape_B, ptrdiff_t *strides_B, char *descr_B,
               int dtype, void *alpha, void *beta)
{
    tblis_tensor A, B;
    _as_tensor(&A, data_A, dtype, ndim_A, shape_A, strides_A, alpha);
    _as_tensor(&B, data_B, dtype, ndim_B, shape_B, strides_B, beta);

    tblis_tensor_add(NULL, NULL, &A, descr_A, &B, descr_B);
}

void tensor_dot(void *data_A, int ndim_A, ptrdiff_t *shape_A, ptrdiff_t *strides_A, char *descr_A,
               void *data_B, int ndim_B, ptrdiff_t *shape_B, ptrdiff_t *strides_B, char *descr_B,
               int dtype, void *result)
{
    tblis_tensor A, B;
    tblis_scalar s;
    _as_tensor(&A, data_A, dtype, ndim_A, shape_A, strides_A, NULL);
    _as_tensor(&B, data_B, dtype, ndim_B, shape_B, strides_B, NULL);

    tblis_tensor_dot(NULL, NULL, &A, descr_A, &B, descr_B, &s);

    ssize_t bytes;
    switch(dtype) {
        case TYPE_SINGLE:
            bytes = sizeof(float);
            break;
        case TYPE_DOUBLE:
            bytes = sizeof(double);
            break;
        case TYPE_SCOMPLEX:
            bytes = sizeof(scomplex);
            break;
        case TYPE_DCOMPLEX:
            bytes = sizeof(dcomplex);
            break;
    }

    memcpy(result, &s.data, bytes);
}
}