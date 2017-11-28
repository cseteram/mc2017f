__kernel void vec_add(__global int4 *A, __global int4 *B, __global int4 *C)
{
    int i = get_global_id(0);
    int4 a = A[i], b = B[i], c;

    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;

    C[i] = c;
}
