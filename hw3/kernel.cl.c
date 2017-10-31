#define TS 64
#define RWPT 4
#define CWPT 4
#define SK 16
#define WIDTH 4

__kernel void mat_mul(
    const __global float *A, const __global float *B, __global float *C,
    const int ROW_A, const int COL_A, const int COL_B)
{
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    float sum = 0.0f;
    for (int k = 0; k < COL_A; ++k) {
        sum += A[i * COL_A + k] * B[k * COL_B + j];
    }
    C[i * COL_B + j] = sum;
}

__kernel void mat_mul_t64(
    const __global float *A, const __global float *B, __global float *C,
    const int ROW_A, const int COL_A, const int COL_B)
{
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = TS * get_group_id(1) + row;
    const int globalCol = TS * get_group_id(0) + col;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float Areg, Breg[CWPT];
    float acc[RWPT][CWPT];

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
                Asub[row + wi][col + wj] = A[(globalRow + wi) * COL_A + (tiledCol + wj)];
                Bsub[row + wi][col + wj] = B[(tiledRow + wi) * COL_B + (globalCol + wj)];
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
                    acc[rw][cw] += Areg * Breg[cw];
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
