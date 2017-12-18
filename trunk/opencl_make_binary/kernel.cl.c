__kernel void vec_add(__global int4 *A, __global int4 *B, __global int4 *C)
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}
