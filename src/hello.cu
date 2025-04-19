#include <stdio.h>
#define N 10000

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void kernel(int* a, int* b, int* c){
    int idx = blockIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // cudamalloc dev_a dev_b dev_c
    HANDLE_ERROR(cudaMalloc((void**) &dev_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**) &dev_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**) &dev_c, sizeof(int) * N));

    // filling the array
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    // cudamemcpy yagesya
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

    kernel<<<N, 1>>>(dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for(int i = 0; i<N; i++){
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return 0;
}
