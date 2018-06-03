#include <stdio.h>
#include <math.h>
#include "cublas_v2.h"

#include "headers/kernels.h"
#include "headers/parameters.h"
#include "headers/boundary_conditions.h"

__constant__ struct SimulationParametes parameters_device;
__constant__ struct Constants constants_device;
__constant__ struct BoundaryInfo boundary_info_device;
__constant__ int coords_device[18];
__constant__ real weights_device[9];
__constant__ int inverse_indices_device[9];


void CUDA_CHECK_ERROR() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void HANDLE_ERROR(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("CUDA STATUS: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void HANDLE_CUBLAS_ERROR(cublasStatus_t stat) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: cublas failed\n");  
        exit(EXIT_FAILURE);
    }
}

void CopyConstantsToDevice(const struct SimulationParametes parameters,
                           const struct Constants constants,
                           const struct BoundaryInfo boundary_info,
                           int *coords,
                           real *weights) {
    HANDLE_ERROR(cudaMemcpyToSymbol(parameters_device,
                                    &parameters,
                                    sizeof(struct SimulationParametes)));

    HANDLE_ERROR(cudaMemcpyToSymbol(constants_device,
                                    &constants,
                                    sizeof(struct Constants)));

    HANDLE_ERROR(cudaMemcpyToSymbol(boundary_info_device,
                                    &boundary_info,
                                    sizeof(struct BoundaryInfo)));

    HANDLE_ERROR(cudaMemcpyToSymbol(coords_device,
                                    coords,
                                    18 * sizeof(int)));

    HANDLE_ERROR(cudaMemcpyToSymbol(weights_device,
                                    weights,
                                    9 * sizeof(real)));
} 

__global__ void CheckConstMemoryCopy() {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id == MASTER) {
        printf(" --- Values allocated in the constant memory--- \n");
        printf(" -> num_lattices %d\n", parameters_device.num_lattices);
        printf(" -> width %d\n", parameters_device.width);

        printf(" -> constant one %f\n", constants_device.one);
        printf(" -> wall velocity %f\n", boundary_info_device.wall_velocity_x);
        printf(" -> weight %f\n\n", weights_device[0]);
    }
}


__device__ int GetIndexDevice(int index_i, int index_j) {
    return index_i + index_j * parameters_device.width; 
}

__global__ void InitArrayDevice(real *array, real init_value, int size) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x; 
    while (thread_id < size) { 
        array[thread_id] = init_value;
        thread_id += blockDim.x * gridDim.x;
    }
}

__global__ void StreamDevice(real *population,
                             real *swap_buffer,
                             int *flag_field) {
    int num_lattices = parameters_device.num_lattices;
    int num_directions = parameters_device.discretization; 
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < num_lattices) {
        if (flag_field[thread_id] == FLUID) {
            for (int component = 0; component < num_directions; ++component) {
                int ii = coords_device[component];
                int jj = coords_device[num_directions + component];
                int neighbour_index = thread_id + GetIndexDevice(ii, jj);
                int shift = component * num_lattices;

                swap_buffer[shift + neighbour_index] = population[shift + thread_id];
            }
        }
        thread_id += blockDim.x * gridDim.x;
    }
}


__global__ void UpdateDensityFieldDevice(real *density,
                                         real *population,
                                         int *flag_field) {
    int num_lattices = parameters_device.num_lattices;
    short int num_directions = parameters_device.discretization; 
    
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < num_lattices) {
        if (flag_field[thread_id] == FLUID) {
            real lattice_density = 0.0;
            for (short int component = 0; component < num_directions; ++component) {
                lattice_density += population[component * num_lattices + thread_id];
            }
            density[thread_id] = lattice_density;
        }
        thread_id += blockDim.x * gridDim.x; 
    }
}

__global__ void UpdateVelocityFieldDevice(real *velocity,
                                          real *population,
                                          real *density,
                                          int *flag_field) {
    int num_lattices = parameters_device.num_lattices;
    short int num_directions = parameters_device.discretization; 
    
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < num_lattices) {
        if (flag_field[thread_id] == FLUID) {
            real lattice_velocity_x = 0.0;
            real lattice_velocity_y = 0.0;
            
            for (short int component = 0; component < num_directions; ++component) {
                real distribution = population[component * num_lattices + thread_id];
                lattice_velocity_x += coords_device[component] * distribution;
                lattice_velocity_y += coords_device[num_directions + component] * distribution;
            }

            real inverse_density = 1.0 / density[thread_id];
            velocity[thread_id] = inverse_density * lattice_velocity_x;
            velocity[num_lattices + thread_id] = inverse_density * lattice_velocity_y;
        }
        thread_id += blockDim.x * gridDim.x; 
    }
}

__global__ void UpdatePopulationFieldDevice(real *velocity,
                                            real *population,
                                            real *density) {
    //int num_lattices = parameters_device.num_lattices;
    short int num_directions = parameters_device.discretization; 

    //real relaxation = parameters_device.relaxation;
    real const_one = constants_device.one;
    real const_two = constants_device.two;
    real const_three = constants_device.three;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < parameters_device.num_lattices) {
        real local_velocity_x = velocity[thread_id];
        real local_velocity_y = velocity[parameters_device.num_lattices + thread_id];

        real dot_product_uu = local_velocity_x * local_velocity_x
                            + local_velocity_y * local_velocity_y;

        for (int component = 0; component < num_directions; ++component) {
            real vector_component_x = coords_device[component];
            real vector_component_y = coords_device[num_directions + component];

            real dot_product_cu = vector_component_x * local_velocity_x
                                + vector_component_y * local_velocity_y;
        
            real velocity_expansion = (const_one * dot_product_cu)
                                    + (const_two * dot_product_cu * dot_product_cu)
                                    - (const_three * dot_product_uu)
                                    + 1.0;

            real equilibrium = weights_device[component]
                             * density[thread_id]
                             * velocity_expansion;

            int shift = component * parameters_device.num_lattices;
            population[shift + thread_id] -= (parameters_device.relaxation
                                           * (population[shift + thread_id] - equilibrium));
        }

        thread_id += blockDim.x * gridDim.x;
    }
} 

__global__ void PrintBC(struct BoundaryConditions *boundary_conditions) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id == MASTER) {
        printf("GPU: num walls %d\n", boundary_conditions->num_wall_elements);
        printf("GPU: num moving walls %d\n", boundary_conditions->num_moving_wall_elements);

    }
}

__global__ void TreatNonSlipBC(int *indices,
                               real *population,
                               int size) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < size) {
        int source = indices[thread_id];
        int target = indices[size + thread_id];
        population[target] = population[source];

        thread_id += blockDim.x * gridDim.x;
    }
}

__global__ void TreatSlipBC(int *indices,
                            real *data,
                            real *density,
                            real *population,
                            int size) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < size) {
        int source = indices[thread_id];
        int target = indices[size + thread_id];
        int index = indices[2 * size + thread_id];

        population[target] = population[source] + data[thread_id] * density[index];

        thread_id += blockDim.x * gridDim.x;
    }
}

__global__ void TreatInflowBC(int *indices,
                              real *data,
                              real *density,
                              real *population,
                              int size) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (thread_id < size) {
        int target = indices[thread_id];
        int index = indices[size + thread_id];


        population[target] = data[thread_id] * density[index];

        thread_id += blockDim.x * gridDim.x;
    }
}

__global__ void TreatOutflowBC(int *indices,
                               real *velocity,
                               real *density,
                               real *population,
                               int size) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int num_lattices = parameters_device.num_lattices;
    int num_directions = parameters_device.discretization;
    
    real const_one = constants_device.one;
    real const_two = constants_device.two;
    real const_three = constants_device.three;

    while (thread_id < size) {
        int source = indices[thread_id];
        int index = indices[size + thread_id];
        int target = indices[2 * size + thread_id];
        int component = indices[3 * size + thread_id];
        
        real neighbour_velocity_x = velocity[index];
        real neighbour_velocity_y = velocity[num_lattices + index];
                
        real i = coords_device[component];
        real j = coords_device[num_directions + component];

        
        real dot_product_uu = neighbour_velocity_x * neighbour_velocity_x
                            + neighbour_velocity_y * neighbour_velocity_y;
        
        real dot_product_cu = -i * neighbour_velocity_x - j * neighbour_velocity_y;
        real dot_product_cu_inv = i * neighbour_velocity_x + j * neighbour_velocity_y;


        real velocity_expansion = (const_one * dot_product_cu)
                                + (const_two * dot_product_cu * dot_product_cu)
                                - (const_three * dot_product_uu)
                                + 1.0;

        real velocity_expansion_inv = (const_one * dot_product_cu_inv)
                                    + (const_two * dot_product_cu_inv * dot_product_cu_inv)
                                    - (const_three * dot_product_uu)
                                    + 1.0;

/*        
        real equilibrium = weights_device[component]
                         * density[index]
                         * velocity_expansion;
        
        real equilibrium_inv = weights_device[component]
                             * density[index]
                             * velocity_expansion_inv;

        
        population[target] = equilibrium
                           + equilibrium_inv
                           - population[source];
*/
        real expansion_sum = velocity_expansion + velocity_expansion_inv;

        population[target] = (weights_device[component] 
                              * density[index]
                              * expansion_sum) 
                           - population[source];


        thread_id += blockDim.x * gridDim.x;
    }
}


__global__ void ComputeVelocityMagnitude(real *velocity,
                                         real *velocity_magnitude) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int size = parameters_device.num_lattices;

    while (thread_id < size) {
        real velocity_x = velocity[thread_id];
        real velocity_y = velocity[size + thread_id];

        velocity_magnitude[thread_id] = sqrt(velocity_x * velocity_x +
                                             velocity_y * velocity_y);
        thread_id += blockDim.x * gridDim.x;
    }

}


__global__ void DrawFluid(uchar4 *ptr,
                          real* velocity_magnitude,
                          int* indices,
                          int size) {
    // map from threadIdx/BlockIdx to pixel position
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = thread_id; i < size; i += stride) {
        int index = indices[thread_id];
        real velocity = velocity_magnitude[index]; 

        // max --> red
        // min --> blue
        // avg --> green
        real max_velocity = parameters_device.max_velocity_rendering;
        real min_velocity = parameters_device.min_velocity_rendering;
        real avg_velocity = (max_velocity + min_velocity) / 2;
        real distance = max_velocity - avg_velocity;

        unsigned int red_hue, blue_hue, green_hue;

        // assuming no negative velocities

        // green slope one -> min -> average
        // green slope two -> average -> max
        real blue_slope = ( - 255 ) / distance;
        real red_slope = ( 255 ) / distance;
        real green_slope_one = (255) / distance;
        real green_slope_two = (-255) / distance;

        int c_blue = -1 * (blue_slope * avg_velocity);
        int c_red = -1 * (red_slope *  avg_velocity);

        int c_green_one = -1 * (green_slope_one * min_velocity);
        int c_green_two = -1 * (green_slope_two * max_velocity);

            if (velocity <= avg_velocity) {

                red_hue = 0;
                blue_hue = blue_slope * velocity + c_blue;
                green_hue = green_slope_one * velocity + c_green_one;  
            }
            else {
                red_hue = red_slope * velocity + c_red;
                blue_hue = 0;
                green_hue = green_slope_two * velocity + c_green_two; 
            }

        red_hue = red_hue <= 255 ? red_hue : 255;
        blue_hue = blue_hue <= 255 ? blue_hue : 255;
        green_hue = green_hue <= 255 ? green_hue : 255;

        ptr[index].x = red_hue;
        ptr[index].y = green_hue;
        ptr[index].z = blue_hue;
        ptr[index].w = 255;
    }
}

__global__ void DrawObstacles(uchar4 *ptr, int* indices, int size) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    while (thread_id < size) {
        int index = indices[thread_id];
        ptr[index].x = 0;
        ptr[index].y = 0;
        ptr[index].z = 0;
        ptr[index].w = 255;

        thread_id += blockDim.x * gridDim.x;
    }
}

__global__ void PrintMaxMinDensity(real *density,
                                   int max_index,
                                   int min_index,
                                   int time) {
    printf("time step: %d; max density: %4.6f; min density: %4.6f\n",
            time, density[max_index], density[min_index]);
}

__global__ void SynchStreams() {
}
