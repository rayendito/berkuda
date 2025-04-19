#include <stdio.h>
#define N 10
#define HEAD 2
#define LD 32
const int threadsPerBlock = LD * LD;

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void kernel(int* a, int* b, int* c_partial){
    __shared__ float cache[threadsPerBlock];
    int flatBlockIdx = blockIdx.x + blockIdx.y * gridDim.x;
    int flatThreadIdx = threadIdx.x + threadIdx.y * blockDim.x;
    int idx = flatThreadIdx + flatBlockIdx * blockDim.x * blockDim.y;
    float temp = 0;

    if(idx < N){
        temp = temp + a[idx] * b[idx];
        idx = idx + blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    }

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 1 && threadIdx.y == 0){
        printf("%d - %d\n", flatThreadIdx, temp);
    }

    cache[flatThreadIdx] = temp;
    __syncthreads();

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        // printf("%d - %d\n", flatThreadIdx, temp);
        // for(int i = 0; i < threadsPerBlock; i++){
        //     printf("%d ", cache[i]);
        // }
    // }

    // while (idx < N){
    //     cache[flatThreadIdx] = a[idx] * b[idx];
    //     idx = idx + blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    // }
}

int main(void){
    // for dot product, c here is a intermediate value for each block
    // we're utilizing device shared memory that's local to every block
    // therefore every thread within a block yagesya
    int *a, *b, *c_partial;
    int *dev_a, *dev_b, *dev_c_partial;

    // let's use the same 2D block and 2D threads format
    // bc hard times create strong men
    dim3 numBlocks(LD, LD, 1);
    dim3 threadsPerBlock(LD, LD, 1);

    // malloc biasa ygy
    a = (int*) malloc(sizeof(int) * N);
    b = (int*) malloc(sizeof(int) * N);
    
    // cache perlu dialloc ga kalo gitu?
    // cache (and dev_c_partial) will be an array yg masih perlu disum (hasil dari tiap block)
    // but it's gonna be small enough that it's reasonable to compute sequentially
    // but how long is it? it's supposed to be as long as how many blocks we have
    int totalBlocks = numBlocks.x * numBlocks.y * numBlocks.z;
    c_partial = (int*) malloc(sizeof(int) * totalBlocks); 

    // filling the array
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    // cudamalloc
    HANDLE_ERROR(cudaMalloc((void**) &dev_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**) &dev_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**) &dev_c_partial, sizeof(int) * totalBlocks));

    // copy to device
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

    kernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c_partial);

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c_partial));

    return 0;
}
