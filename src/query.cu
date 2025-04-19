
#include <stdio.h>

int main(void){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("device count %d\n",deviceCount);

    for(int i = 0; i < deviceCount; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d:  %s\n", i, prop.name);
        printf("        Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
        printf("        Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("        Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("        Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("        Max Threads Dim: (%d, %d, %d)\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("        Max Grid Size: (%d, %d, %d)\n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");

    }

    return 0;
}
