#define TS 64
#define RWPT 4
#define CWPT 4
#define SK 16
#define WIDTH 4

__kernel void transpose(
    __global uchar *A,
    __global uchar *B,
    const int P, const int Q)
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    
    int gi = get_group_id(1), gj = get_group_id(0);
    int li = get_local_id(1), lj = get_local_id(0);

    __local uchar X[16][16];

    if (i < P && j < Q) {
        X[li][lj] = A[i * Q + j];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    int ni = gj * 16 + li;
    int nj = gi * 16 + lj;
    if (ni < Q && nj < P) {
        B[ni * P + nj] = (int)X[lj][li];
    }
}

__kernel void conv(
    const __global uchar *A,
    const __global uchar *B,
    __global int *C,
    const int ROW_A, const int COL_A, const int COL_B)
{
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = TS * get_group_id(1) + row;
    const int globalCol = TS * get_group_id(0) + col;

    __local int Asub[TS][TS];
    __local int Bsub[TS][TS];

    int Areg, Breg[CWPT];
    int acc[RWPT][CWPT];

    #pragma unroll
    for (int rw = 0; rw < RWPT; rw++) {
        #pragma unroll
        for (int cw = 0; cw < CWPT; cw++) {
            acc[rw][cw] = 0.0f;
        }
    }

    const int numTiles = COL_A >> 6;
    int t = 0;
    do {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;

        #pragma unroll
        for (int rw = 0; rw < RWPT; rw++) {
            #pragma unroll
            for (int cw = 0; cw < CWPT; cw++) {
                int wi = SK * rw, wj = SK * cw;
                Asub[row + wi][col + wj] = (int)A[(globalRow + wi) * COL_A + (tiledCol + wj)];
                Bsub[row + wi][col + wj] = (int)B[(tiledRow + wi) * COL_B + (globalCol + wj)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TS; k++) {
            #pragma unroll
            for (int cw = 0; cw < CWPT; cw++) {
                Breg[cw] = Bsub[k][col + SK * cw];
            }

            #pragma unroll
            for (int rw = 0; rw < RWPT; rw++) {
                Areg = Asub[row + SK * rw][k];
                #pragma unroll
                for (int cw = 0; cw < CWPT; cw++) {
                    // acc[rw][cw] += (Areg - Breg[cw]) * (Areg - Breg[cw]);
                    int x = Areg - Breg[cw];
                    acc[rw][cw] += mul24(x, x);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        t++;
    } while (t < numTiles);

    #pragma unroll
    for (int rw = 0; rw < RWPT; rw++) {
        #pragma unroll
        for (int cw = 0; cw < CWPT; cw++) {
            int wi = SK * rw, wj = SK * cw;
            C[(globalRow + wi) * COL_B + (globalCol + wj)] = acc[rw][cw];
        }
    }
}

__kernel void reduction(
    __global int *diff,
    __global int *min_diff,
    __global int *idx,
    __local int *l_min_diff,
    __local int *l_idx,
    const int num_tiles, const int num_filters, const int padd_num_filters)
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    int lj = get_local_id(0);

    l_min_diff[lj] = (j < num_filters) ? diff[i * padd_num_filters + j] : INT_MAX;
    l_idx[lj] = j;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
        if (lj < p && l_min_diff[lj + p] < l_min_diff[lj]) {
            l_min_diff[lj] = l_min_diff[lj + p];
            l_idx[lj] = l_idx[lj + p];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lj == 0) {
        int index = i * get_num_groups(0) + get_group_id(0);
        min_diff[index] = l_min_diff[0];
        idx[index] = l_idx[0];
    }
}
