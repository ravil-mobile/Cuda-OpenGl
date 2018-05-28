#include "headers/helper_cuda.h"

void ChooseGPU() {

    // choose a proper gpu device with max compute capability
    cudaDeviceProp property;
    int gpu_count = 0;
    int gpu_device = 0;
    int max_compute_capability = 0;

    HANDLE_ERROR(cudaGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        printf("ERROR: no gpu has been detected on your system\n");
        exit(EXIT_FAILURE);
    }
    
    printf("number of gpu devices detected: %d\n\n", gpu_count);

    for (int gpu_instance = 0; gpu_instance < gpu_count; ++gpu_instance) {
        HANDLE_ERROR(cudaGetDeviceProperties(&property, gpu_instance));

//#ifdef DEBUG
        printf(" --- General Information for device %d ---\n", gpu_instance);
        printf("\t name: %s\n", property.name);
        printf("\t compute capability %d.%d\n", property.major, property.minor);
        printf("\t warp size: %d\n", property.warpSize);
        printf("\t max num. threads per block: %d\n", property.maxThreadsPerBlock);
        printf("\t max num. registers per block: %d\n", property.regsPerBlock);
        printf("\t size of constant memory: %lu\n", property.totalConstMem);
        printf("\t The number of blocks allowed along x dimension of a grid: %d\n\n\n",
               property.maxGridSize[0]);
//#endif
        int factor = 10;
        int compute_capability = factor * property.major + property.minor;
        if (compute_capability > max_compute_capability) {
            gpu_device = gpu_instance;
            max_compute_capability = compute_capability;
        }
    } 

    HANDLE_ERROR(cudaSetDevice(gpu_device));
    HANDLE_ERROR(cudaGetDeviceProperties(&property, gpu_device));
    
    printf("\n --- %s: device has been chosen --- \n", property.name);
    printf(" --- Compute capability: %d.%d --- \n\n\n", property.major, property.minor);
}
