//#ifdef GRAPHICS
//#define GLEW_STATIC
    #include<GL/glew.h>
    #include<GLFW/glfw3.h>
    #include<cuda_gl_interop.h>
//#endif

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <argp.h>

#include "headers/parameters.h"
#include "headers/scanning.h"
#include "headers/boundary_conditions.h"
#include "headers/helper.h"
#include "headers/kernels.h"
#include "headers/domain.h"
#include "headers/helper_cuda.h"

const int SCR_HEIGHT = 630;
const int SCR_WIDTH = 930; 

// process key inputs to close window
void processInput (GLFWwindow* window) {
	// if the escape key has been pressed
	// otherwise glfwGetKey returns GFLW_RELEASE
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose( window, true );
	}
}

// Create frame resize callback function
void framebuffer_size_callback( GLFWwindow* window, int height, int width) {
	glViewport(0,0, width, height);
}


// global variables for cuda interop
GLuint buffer_object;
cudaGraphicsResource * resource;


int main(int argc, char **argv) {

    // read input arguments
    static struct argp_option options[] = {
        {"parameters",  'p', "FILE", 0, "Sets path to a parameter file"},
        {"boundary-cond", 'b', "FILE", 0, "Sets path to a boundary conditions file"},
        {"mesh", 'm', "FILE", 0, "Sets path to a mesh file"},
        {"threads-distr", 't', "FILE", 0, "Sets path to a threads distribution file"},
        { 0 }
    };

    struct Arguments arguments;
    arguments.parameters_file = NULL;
    arguments.boundary_cond_file = NULL;
    arguments.mesh_file = NULL;
    arguments.threads_distr_file = NULL;
    
    static char doc[] = "lbm -- 2D implementation of the Lattice Boltzmann Method in CUDA";
    static char args_doc[] = "--parameters=FILE --boundary-cond=FILE --mesh=FILE --threads-distr=FILE";

    static struct argp argp = {options, parse_opt, args_doc, doc};
    argp_parse (&argp, argc, argv, 0, 0, &arguments);


    ChooseGPU();
       
    // read input data
    struct SimulationParametes parameters;
    ReadParameterFile(arguments.parameters_file, parameters);

    struct Constants constants;
    constants.one = 3.0;
    constants.two = 4.5;
    constants.three = 1.5;

    struct BoundaryInfo boundary_info;
    ReadBoundaryFile(arguments.boundary_cond_file, boundary_info);

    // allocate constant data into the DEVICE
    CopyConstantsToDevice(parameters,
                          constants,
                          boundary_info,
                          coords,
                          weights);

#ifdef DEBUG    
    CheckConstMemoryCopy<<<1,1>>>();
    cudaDeviceSynchronize();
#endif

    // allocate memory on the HOST
    int *flag_field = (int*)calloc(parameters.num_lattices, sizeof(int));
    ReadMeshFile(arguments.mesh_file, flag_field, parameters);

    // allocate and init DOMAIN on the DEVICE
    DomainHandler domain_handler;
    domain_handler.InitDomainOnDevice(parameters,
                                      flag_field);

    const Domain *domain = domain_handler.GetDeviceData();

    // allocate and init BOUNDARIES on the DEVICE
    BoundaryConditionsHandler bc_handler;
    
    ScanFlagField(flag_field,
                  bc_handler,
                  parameters,
                  constants,
                  boundary_info);

    const BoundaryConditions *boundaries = bc_handler.GetDeviceData();

    CudaResourceDistr threads_distr;
    ReadThreadsDistrFile(arguments.threads_distr_file, threads_distr);

    CudaResourceDistr blocks_distr;
    ComputeBlcoksDistr(blocks_distr, threads_distr, parameters, boundaries); 
    
    int threads = 0;
    int blocks = 0;

#ifdef GRAPHICS
    // set OpenGL device
    glfwInit();

    // Create window 
    GLFWwindow* window = glfwCreateWindow (SCR_WIDTH, SCR_HEIGHT, "Lattice Boltzmann", NULL, NULL);

    if (window == NULL) {
        std::cout << "ERROR: failed to create window." << std::endl; 
        glfwTerminate();
        exit(EXIT_FAILURE); 
    }

    // Bind object to context
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cout << "ERROR: failed to initialize glew" << std::endl; 
        exit(EXIT_FAILURE);
    }
    
    // first two paramters set location of left corner, other two are width and height	
    glViewport(0,0, SCR_WIDTH, SCR_HEIGHT);
	
    // Register call back function with glfw
    glfwSetFramebufferSizeCallback( window, framebuffer_size_callback);

    // create buffer object to hold pixel data
    glGenBuffers(1, &buffer_object);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_object);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, parameters.num_lattices * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    // register buffer with cuda runtime
    cudaGraphicsGLRegisterBuffer(&resource, buffer_object, cudaGraphicsMapFlagsNone);

    // uchar4 is defined by cuda
    uchar4* dev_ptr;
    size_t size; 
#endif

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // START of algorighm
    cudaEventRecord(start, 0);
    for (int time = 0; (time < parameters.num_time_steps); ++time) {
        
        // perform streaming step
        threads = threads_distr.stream_device;
        blocks = blocks_distr.stream_device;
        StreamDevice<<<blocks, threads>>>(domain->dev_population,
                                          domain->dev_swap_buffer,
                                          domain->dev_flag_field);
        CUDA_CHECK_ERROR(); 
        
        // apply boundary consitions
        if (boundaries->num_wall_elements != 0) {
            threads = threads_distr.treat_non_slip_bc;
            blocks = blocks_distr.treat_non_slip_bc;
            TreatNonSlipBC<<<blocks, threads>>>(boundaries->bc_wall_indices,
                                                domain->dev_swap_buffer,
                                                boundaries->num_wall_elements); 
            CUDA_CHECK_ERROR();
        }

        if (boundaries->num_moving_wall_elements != 0) {
            threads = threads_distr.treat_slip_bc;
            blocks = blocks_distr.treat_slip_bc;
            TreatSlipBC<<<blocks, threads>>>(boundaries->bc_moving_wall_indices,
                                         boundaries->bc_moving_wall_data,
                                         domain->dev_density,
                                         domain->dev_swap_buffer,
                                         boundaries->num_moving_wall_elements);
            CUDA_CHECK_ERROR();
        }

        if (boundaries->num_inflow_elements != 0) {
            threads = threads_distr.treat_inflow_bc;
            blocks = blocks_distr.treat_inflow_bc;
            TreatInflowBC<<<blocks, threads>>>(boundaries->bc_inflow_indices,
                                               boundaries->bc_inflow_data,
                                               domain->dev_density,
                                               domain->dev_swap_buffer,
                                               boundaries->num_inflow_elements);
            CUDA_CHECK_ERROR();
        }

        if (boundaries->num_outflow_elements != 0) {
            threads = threads_distr.treat_outflow_bc;
            blocks = blocks_distr.treat_outflow_bc;
            TreatOutflowBC<<<blocks, threads>>>(boundaries->bc_outflow_indices,
                                                domain->dev_velocity,
                                                domain->dev_density,
                                                domain->dev_swap_buffer,
                                                boundaries->num_outflow_elements);
            CUDA_CHECK_ERROR();
        }


        HANDLE_ERROR(cudaDeviceSynchronize());
        domain_handler.SwapPopulationFields(); 
       
        // perform collision step
        threads = threads_distr.update_density_field_device;
        blocks = blocks_distr.update_density_field_device;
        UpdateDensityFieldDevice<<<blocks, threads>>>(domain->dev_density,
                                                      domain->dev_population,
                                                      domain->dev_flag_field);

        CUDA_CHECK_ERROR(); 

        threads = threads_distr.update_velocity_field_device;
        blocks = blocks_distr.update_velocity_field_device;
        UpdateVelocityFieldDevice<<<blocks, threads>>>(domain->dev_velocity,
                                                       domain->dev_population,
                                                       domain->dev_density,
                                                       domain->dev_flag_field);
        CUDA_CHECK_ERROR(); 
        
        
        threads = threads_distr.update_population_field_device;
        blocks = blocks_distr.update_population_field_device;
        UpdatePopulationFieldDevice<<<blocks, threads>>>(domain->dev_velocity,
                                                         domain->dev_population,
                                                         domain->dev_density);
        
        CUDA_CHECK_ERROR();

#ifdef DEBUG
        if ((time % parameters.steps_per_report) == 0) {
            int max_index = 0;
            int min_index = 0;
            HANDLE_CUBLAS_ERROR(cublasIdamax(handle,
                                             parameters.num_lattices,
                                             domain->dev_density,
                                             1,
                                             &max_index));

            HANDLE_CUBLAS_ERROR(cublasIdamin(handle,
                                             parameters.num_lattices,
                                             domain->dev_density,
                                             1,
                                             &min_index));

            real max_density = 0.0;
            real min_density = 0.0;

            // copy data from device to host to print using std::cout
            HANDLE_ERROR(cudaMemcpy(&max_density,
                         domain->dev_density + max_index - 1,
                         sizeof(real),
                         cudaMemcpyDeviceToHost));

            HANDLE_ERROR(cudaMemcpy(&min_density,
                         domain->dev_density + min_index - 1,
                         sizeof(real),
                         cudaMemcpyDeviceToHost));

            std::cout << "time step: " << time << "; ";
            std::cout << "max density: " << max_density << "; ";
            std::cout << "min density "  << min_density << std::endl;
        }
#endif

#ifdef GRAPHICS
            threads = threads_distr.compute_velocity_magnitude;
            blocks = blocks_distr.compute_velocity_magnitude;
            ComputeVelocityMagnitude<<<blocks, threads>>>(domain->dev_velocity,
                                                          domain->dev_velocity_magnitude);

            cudaGraphicsMapResources(1, &resource, NULL);
            cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);

            processInput(window);
            
            threads = threads_distr.float_to_rgb;
            blocks = blocks_distr.float_to_rgb;
            FloatToRGB<<<blocks, threads>>>(dev_ptr,
                                            domain->dev_velocity_magnitude,
                                            domain->dev_flag_field);

          // unmap resources to synchronize between rendering and cuda tasks 
          cudaGraphicsUnmapResources(1, &resource, NULL);

          glDrawPixels(parameters.width, parameters.height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
          glfwSwapBuffers(window);
          glfwPollEvents(); 
#endif
    
    }
    // END of algorithm
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop); 
    
    double MLUPS = (parameters.num_lattices * parameters.num_time_steps)
                 / (elapsed_time * 1e3);

    printf("MLUPS: %4.6f\n", MLUPS);

    // free HOST recourses
    cudaGraphicsUnregisterResource(resource);
    glfwTerminate();
    cublasDestroy(handle);
    
    // free DEVICE resources
    free(flag_field);
    return 0;
}
