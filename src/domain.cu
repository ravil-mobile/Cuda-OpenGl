#include <stdio.h>
#include <algorithm>
#include <vector>

#include "headers/domain.h"
#include "headers/parameters.h"
#include "headers/kernels.h"
#include "headers/helper.h"

DomainHandler::DomainHandler() {
    dev_domain.dev_flag_field = NULL;
    dev_domain.dev_density = NULL;
    dev_domain.dev_velocity_magnitude = NULL;
    dev_domain.dev_velocity = NULL;
    dev_domain.dev_population = NULL;
    dev_domain.dev_swap_buffer = NULL;

    dev_domain.dev_fluid_indices = NULL;
    dev_domain.dev_solid_indices = NULL;
}

DomainHandler::~DomainHandler() {
    if (dev_domain.dev_flag_field != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_flag_field));
    }
        
    if (dev_domain.dev_density != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_density));
    }
        
    if (dev_domain.dev_velocity != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_velocity));
    }

    if (dev_domain.dev_velocity_magnitude != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_velocity_magnitude));
    }

    if (dev_domain.dev_population != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_population));
    }

    if (dev_domain.dev_swap_buffer != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_swap_buffer));
    }

    if (dev_domain.dev_fluid_indices != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_fluid_indices)); 
    }

    if (dev_domain.dev_solid_indices != NULL) {
        HANDLE_ERROR(cudaFree(dev_domain.dev_solid_indices));
    }
}

void DomainHandler::InitDomainOnDevice(SimulationParametes &parameters,
                                       int *flag_field) {

    int size = parameters.num_lattices;
    int dimension = parameters.dimension;
    int discretization = parameters.discretization;

    // allocate Memory on the DEVICE
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_flag_field), size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_density), size * sizeof(real)));
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_velocity), dimension * size * sizeof(real)));
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_velocity_magnitude), size * sizeof(real)));
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_population), discretization * size * sizeof(real)));
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_swap_buffer), discretization * size * sizeof(real)));

    // init memory in the DEVICE
    HANDLE_ERROR(cudaMemcpy(dev_domain.dev_flag_field,
                            flag_field,
                            size * sizeof(int),
                            cudaMemcpyHostToDevice));

    double init_value = 1.0;

    const int num_threads = 128;
    int num_blocks = ComputeNumBlocks(num_threads, size);
    InitArrayDevice<<<num_blocks, num_threads>>>(dev_domain.dev_density,
                                                 init_value,
                                                 size); 

    init_value = 0.0;
    InitArrayDevice<<<num_blocks, num_threads>>>(dev_domain.dev_velocity,
                                                 init_value,
                                                 dimension * size);
    
    InitArrayDevice<<<num_blocks, num_threads>>>(dev_domain.dev_velocity_magnitude,
                                                 init_value,
                                                 size);


    for (int i = 0; i < discretization; ++i) {

        init_value = weights[i];
        InitArrayDevice<<<num_blocks, num_threads>>>((dev_domain.dev_population + i * size),
                                                     init_value,
                                                     parameters.num_lattices);

        InitArrayDevice<<<num_blocks, num_threads>>>((dev_domain.dev_swap_buffer + i * size),
                                                     init_value,
                                                     parameters.num_lattices);
    } 
    
    HANDLE_ERROR(cudaDeviceSynchronize());
}

void DomainHandler::AllocateFluidElementIndices(std::vector<int> indices) {
    if (dev_domain.dev_fluid_indices != NULL) {
        HANDLE_ERROR( cudaFree (dev_domain.dev_fluid_indices));
    }
    dev_domain.num_fluid_elements = indices.size();
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_fluid_indices), indices.size() * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_domain.dev_fluid_indices,
                            indices.data(),
                            indices.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
}


void DomainHandler::AllocateSolidElementIndices(std::vector<int> indices) {
    if (dev_domain.dev_solid_indices != NULL) {
        HANDLE_ERROR( cudaFree (dev_domain.dev_solid_indices));
    }
    dev_domain.num_solid_elements = indices.size();
    HANDLE_ERROR(cudaMalloc(&(dev_domain.dev_solid_indices), indices.size() * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_domain.dev_solid_indices,
                            indices.data(),
                            indices.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
}

void DomainHandler::UpdateFlagField (int* flag_field, unsigned int size) {

    HANDLE_ERROR( cudaMemcpy(dev_domain.dev_flag_field,
                             flag_field,
                             size * sizeof(int),
                             cudaMemcpyHostToDevice ) );
}

void DomainHandler::SwapPopulationFields() {
    std::swap(dev_domain.dev_population,
              dev_domain.dev_swap_buffer);
}

const Domain * DomainHandler::GetDeviceData() {
    return &dev_domain;
}

