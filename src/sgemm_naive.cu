#include <stdio.h>
#define BLOCKDIM 32
#define CEIL_DIV(A, B) ((A + B-1) / B)

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// sgemm stands for Single precision GEneral Matrix Multiply. according to the BLAS libraries, it computes the following:
// C = alpha * A * B + beta * C
// where
// A is a matrix of dimension M x K
// B is ------//------------- K x N
// C is ------//------------- M x N
// alpha and beta are constants. note: setting alpha = 1 and beta = 0 is just a matrix multiplication A * B

// sgemm naive does one "entry" of the resulting matrix C sequentially still by each thread.
// that's why the number of threads launched is as much as the resulting matrix (M x N, split into blocks of threads)

// C is what we're writing to so it hasn't to be a constant
__global__ void sgemm_naive(
    int M, int N, int K,
    const float * A, const float * B, float * C,
    float alpha, float beta
) {
    // which "coordinate" of C this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N){ // to account for "overlaunching" threads
        // add sequentially
        float tmp = 0.0;
        for(int i = 0; i < K; i ++){
            tmp += A[x * K + i] * B[i * N + y];
        }
        
        // write to C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main(){
    int M = 5;
    int N = 3;
    int K = 3;
    float alpha = 2.0;
    float beta = 0.0;

    dim3 gridDim(CEIL_DIV(M, BLOCKDIM), CEIL_DIV(N, BLOCKDIM), 1);
    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);
    
    // initializing matrices for host
    float dummy_A[M * K] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    };
    float dummy_B[K * N] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };

    float * A = (float *) malloc(sizeof(float) * M * K);
    float * B = (float *) malloc(sizeof(float) * K * N);
    float * C = (float *) malloc(sizeof(float) * M * N); // even though it's not used *now*
    
    for(int i = 0; i < M*K; ++i) A[i] = dummy_A[i];
    for(int i = 0; i < K*N; ++i) B[i] = dummy_B[i];
    
    // matrices for device and data tranfers
    float *dev_A, *dev_B, *dev_C;
    HANDLE_ERROR(cudaMalloc(&dev_A, sizeof(float) * M * K));
    HANDLE_ERROR(cudaMalloc(&dev_B, sizeof(float) * K * N));
    HANDLE_ERROR(cudaMalloc(&dev_C, sizeof(float) * M * N));

    HANDLE_ERROR(cudaMemcpy(dev_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    sgemm_naive<<<blockDim, gridDim>>>(M, N, K, dev_A, dev_B, dev_C, alpha, beta);

    // copy results back to host
    HANDLE_ERROR(cudaMemcpy(C, dev_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    // print the matrix
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            printf("%.2f   ", C[i * N + j]);
        }
        printf("\n");
    }


    return 0;
}