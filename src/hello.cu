#include <stdio.h>
#define N 1000000
#define HEAD 10

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void kernel(int* a, int* b, int* c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < N){
        c[idx] = a[idx] + b[idx];
        idx = idx + blockDim.x * gridDim.x;
    }
}

int main(void){
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    dim3 numBlocks(20, 1, 1);
    dim3 threadsPerBlock(1024, 1, 1);

    // malloc biasa ygy
    a = (int*) malloc(sizeof(int) * N);
    b = (int*) malloc(sizeof(int) * N);
    c = (int*) malloc(sizeof(int) * N);

    // filling the array
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    // cudamalloc dev_a dev_b dev_c biar allocate di heap, not di stack
    HANDLE_ERROR(cudaMalloc((void**) &dev_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**) &dev_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**) &dev_c, sizeof(int) * N));

    // cudamemcpy yagesya
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

    kernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for(int i = 0; i<HEAD; i++){
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    }

    printf("...\n");
    for(int i = N-HEAD; i<N; i++){
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return 0;
}
