#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map> 
#include <algorithm>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <argp.h>
#include <sstream>
#include "headers/parameters.h"
#include "headers/helper.h"
#include "headers/boundary_conditions.h"

int GetIndex(int index_i,
             int index_j,
             const int width) {
    return (index_i + index_j * width);
}

const int NUM_OF_PARAMETERS = 4;

int parse_opt(int key, char *arg, struct argp_state *state) {
    struct Arguments *arguments = (struct Arguments*)state->input;
    static int flags[NUM_OF_PARAMETERS] = {0, 0, 0, 0};

    switch (key) {
        case 'p':
            arguments->parameters_file = arg;
            flags[0] = 1;
            break;

        case 'b':
            arguments->boundary_cond_file = arg;
            flags[1] = 1;
            break;

        case 'm':
            arguments->mesh_file = arg;
            flags[2] = 1;
            break;

        case 't':
            arguments->threads_distr_file = arg;
            flags[3] = 1;
            break;

        case ARGP_KEY_END:
            int counter = 0;
            for (int i = 0; i < NUM_OF_PARAMETERS; ++i) {
                counter += flags[i];
            }

            if (counter < 4) {
                printf("ERROR: please, read --usage or --help to get more information\n");
                argp_failure (state, 1, 0, "too few arguments");
            } else if (counter < 4) {
                printf("ERROR: please, read --usage or --help to get more information\n");
                argp_failure (state, 1, 0, "too many arguments");
            }
            break;
    }
    return 0;
}


bool isSpace (const std::string& str) {
    return str.find_first_not_of(' ') == str.npos;
}

bool predicate(const std::string& line) {
    std::string delimiter = "=";
    if(line.compare(0, 1, delimiter) || line.empty() || isSpace(line)) {
        return false;
    }

    return true;
}

void PrintErrorMessage(std::string fault) {
    std::cout << "ERROR: parameter ["
              << fault 
              << "] is not specified in the file "
              << "or it cannot be read"
              << std::endl
              << "HINT: please, refer to the spec"
              << std::endl;
}

void ReadParameterFile(const char* parameter_file,
                       struct SimulationParametes &parameters) {
    /*
        This function reads data from the parameter file
        and populates the parameter struct. Only parameters
        defined in the two maps below are read from the
        file. Everything else must be specified at the
        end of the function.
        Ergo, we don't just read the data from the file,
        but also initilize parameters not defined in the file.
        Any lines starting with # will be ignored. 

        Example of a file:

        ####################################################
        # HEADER: the file desctibes simulation parameters #
        ####################################################
 
        =
        num_time_steps 200000000
        width 930
        height 630

        delta_x 0.000577
        delta_t 1e-3

        tau 0.6
        viscosity 85.06e-6

        max_velocity_rendering 0.125
        min_velocity_rendering 0.00000001
        steps_per_report 100

  */

    std::ifstream input_file;
    if (parameter_file != NULL) {
        input_file.open(parameter_file);
    }
    else {
        std::cout << "ERROR: parameter file hasn't been provided" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!input_file) {
        std::cout << "File [" << parameter_file << "] could not be opened";
        exit(EXIT_FAILURE);
    }

    std::string input;
    std::vector<std::string> lines;

    // Read lines into vector
    while(std::getline(input_file, input))   {
        lines.push_back(input);
    }

    input_file.close();

    // Trim lines
    std::vector<std::string>::iterator end = std::remove_if(lines.begin(), lines.end(), predicate);

    // Process file to extract data
    std::map<std::string, int> intMap;
    std::map<std::string, real> doubleMap;

    // initilize maps
    const real DEFAULT_VALUE = 0.0;

    intMap["num_time_steps"] = DEFAULT_VALUE;
    intMap["width"] = DEFAULT_VALUE;
    intMap["height"] = DEFAULT_VALUE;
    intMap["steps_per_report"] = DEFAULT_VALUE;
    intMap["brush_size"] = DEFAULT_VALUE;

    doubleMap["delta_x"] = DEFAULT_VALUE;
    doubleMap["delta_t"] = DEFAULT_VALUE;
    doubleMap["viscosity"] = DEFAULT_VALUE;
    doubleMap["tau"] = DEFAULT_VALUE;
    doubleMap["max_velocity_rendering"] = DEFAULT_VALUE;
    doubleMap["min_velocity_rendering"] = DEFAULT_VALUE;

    // populate data
    for (std::vector<std::string>::iterator i = lines.begin(); i != end; i++) {
        std::map<std::string, int>::iterator intIt;
        std::map<std::string, real>::iterator doubleIt;
        std::stringstream stream (*i);
        std::string key;
        real value;

        stream >> key >> value;
        if((intIt = intMap.find(key)) != intMap.end()) {
          intIt->second = (int)value;
        }
        else if ((doubleIt = doubleMap.find(key)) != doubleMap.end()) {
            doubleIt->second = value; 
        }
    }

    // check whether all necessary parameters are specidied within the file
    for (std::map<std::string, int>::iterator it = intMap.begin();
         it != intMap.end();
         ++it) {
        if (it->second == DEFAULT_VALUE) {
            PrintErrorMessage(it->first);
            exit(EXIT_FAILURE);
        }    
    }
    
    for (std::map<std::string, real>::iterator it = doubleMap.begin();
         it != doubleMap.end();
         ++it) {
        if (it->second == DEFAULT_VALUE) {
            PrintErrorMessage(it->first);
            exit(EXIT_FAILURE);
        }    
    }

    // assigned values to the structure
    parameters.num_time_steps = intMap["num_time_steps"];
    parameters.width = intMap["width"];
    parameters.height = intMap["height"];
    parameters.steps_per_report = intMap["steps_per_report"];
    parameters.brush_size = intMap["brush_size"];

    parameters.delta_x = doubleMap["delta_x"];
    parameters.delta_t = doubleMap["delta_t"];
    parameters.viscosity = doubleMap["viscosity"];
    parameters.tau = doubleMap["tau"];
    parameters.max_velocity_rendering = doubleMap["max_velocity_rendering"];
    parameters.min_velocity_rendering = doubleMap["min_velocity_rendering"];

    // Initialize the rest of the struct
    parameters.simulation_time = 1.0;
    parameters.dimension = 2;
    parameters.discretization = 9;
    parameters.speed_of_sound = 1.0 / sqrt(3.0);
    parameters.relaxation = 1.0 / parameters.tau;
    parameters.num_lattices = parameters.width * parameters.height;
}

void ReadBoundaryFile(const char* boundary_file,
                      struct BoundaryInfo &boundary_info) {
    /*
        Read file containing boundary values. Specifies default values
        for parameters that are not specified in the file. 

        Example of a file:
        
        ##################################################
        # HEADER: the file desctibes boundary conditions #
        ##################################################
 
        =
        wall_velocity_x 0.00
        wall_velocity_y 0.00

        velocity_inflow_x 0.15
        velocity_inflow_y 0.00

        density_outflow 1.0

    */

    std::ifstream input_file;
    if (boundary_file != NULL) {
        input_file.open(boundary_file);
    }
    else {
        std::cout << "ERROR: file with boundary consitions hasn't been provided" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!input_file) {
        std::cout << "File [" << boundary_file << "] could not be opened";
        exit(EXIT_FAILURE);
    }


    std::vector<std::string> lines;
    std::string line;

    while (getline(input_file, line))  {
      lines.push_back(line);
    }

    input_file.close();

    // Trim lines
    std::vector<std::string>::iterator end = std::remove_if(lines.begin(), lines.end(), predicate);

    // Process file to extract data
    std::map<std::string, real> doubleMap;

    // specify default values
    real DEFAULT_VALUE = -999.999;
    doubleMap["wall_velocity_x"] = DEFAULT_VALUE;
    doubleMap["wall_velocity_y"] = DEFAULT_VALUE;
    doubleMap["velocity_inflow_x"] = DEFAULT_VALUE;
    doubleMap["velocity_inflow_y"] = DEFAULT_VALUE;
    doubleMap["density_outflow"] = DEFAULT_VALUE;

    // populate data
    for (std::vector<std::string>::iterator i = lines.begin(); i != end; i++) {
        std::map<std::string, real>::iterator doubleIt;
        std::stringstream stream (*i);
        std::string key;
        real value;

        stream >> key >> value;
        if ((doubleIt = doubleMap.find(key)) != doubleMap.end()) {
            doubleIt->second = value;
        }

    }
    
    for (std::map<std::string, real>::iterator it = doubleMap.begin();
         it != doubleMap.end();
         ++it) {
        if (it->second == DEFAULT_VALUE) {
            PrintErrorMessage(it->first);
            exit(EXIT_FAILURE);
        }    
    }
    
    boundary_info.wall_velocity_x = doubleMap["wall_velocity_x"];
    boundary_info.wall_velocity_y = doubleMap["wall_velocity_y"];
    boundary_info.velocity_inflow_x = doubleMap["velocity_inflow_x"];
    boundary_info.velocity_inflow_y = doubleMap["velocity_inflow_y"];
    boundary_info.density_outflow = doubleMap["density_outflow"];
}

void ReadMeshFile(const char *mesh_file,
                  int *flag_field,
                  struct SimulationParametes &parameters) {
    /* Populate flag_field from mesh_file 
    */
    
    std::ifstream input_file;
    if (mesh_file != NULL) {
        input_file.open(mesh_file);
    }

    else {
        std::cout << "ERROR: mesh file hasn't been provided" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!input_file) {
        std::cout << "ERROR: file [" << mesh_file << "] could not be opened" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string input;
    std::vector<std::string> lines;

    // read file into vector
    while( std::getline(input_file, input)) {
        lines.push_back(input);
    }

    input_file.close();
    
    // extract ints from lines
    unsigned int count = 0;
    for (std::vector<std::string>::iterator it = lines.begin();
       it != lines.end(); it++) {
        std::stringstream curr_line (*it);
        std::vector<int> flag_line;
        int temp;
        while (curr_line >> temp ) {
            flag_line.push_back(temp);
        }

        if (count > parameters.height) {
            std::cout << "ERROR: mesh file: [" 
                      << mesh_file 
                      << "] inconsistent with the file of parameters"
                      << "in terms of [height] of the domain"
                      << std::endl;

            exit(EXIT_FAILURE);
        }

        if (flag_line.size() > parameters.width) {
            std::cout << "ERROR: mesh file: [" 
                      << mesh_file 
                      << "] inconsistent with the file of parameters"
                      << "in terms of [width] of the domain"
                      << std::endl;

            exit(EXIT_FAILURE);
        }


        std::copy(flag_line.begin(), flag_line.end(), flag_field + count * flag_line.size());
        count += 1;
    }
}

void ReadThreadsDistrFile(char *threads_distr_file,
                          struct CudaResourceDistr &threads_distr) {
    /*
    Example of a file:
    
        ############################################################################
        # HEADER: the file desctibes theads distriburion per block for each kernel #
        ############################################################################
     
        =
        stream_device                   192
        update_density_field_device     192
        update_velocity_field_device    192
        update_population_field_device  192

        treat_non_slip_bc               192
        treat_slip_bc                   192
        treat_inflow_bc                 192
        treat_outflow_bc                192

        compute_velocity_magnitude      192
        float_to_rgb                    192
    */
    std::ifstream input_file;
    if (threads_distr_file != NULL) {
        input_file.open(threads_distr_file);
    }

    else {
        std::cout << "ERROR: thread distribution file hasn't been provided" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!input_file) {
        std::cout << "ERROR: file [" << threads_distr_file << "] could not be opened" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string input;
    std::vector<std::string> lines;

    // read file into vector
    while(std::getline(input_file, input)) {
        lines.push_back(input);
    }

    input_file.close();
    // Trim lines
    std::vector<std::string>::iterator end = std::remove_if(lines.begin(), lines.end(), predicate);
    
    // Process file to extract data
    std::map<std::string, int> intMap;
    
    // initilize maps
    // initilize maps                                                                                      
    const int DEFAULT_VALUE = 0;                                                                        

    intMap["stream_device"] = DEFAULT_VALUE;                                                              
    intMap["update_density_field_device"] = DEFAULT_VALUE;
    intMap["update_velocity_field_device"] = DEFAULT_VALUE;
    intMap["update_population_field_device"] = DEFAULT_VALUE;
    intMap["treat_non_slip_bc"] = DEFAULT_VALUE;    
    intMap["treat_slip_bc"] = DEFAULT_VALUE;         
    intMap["treat_inflow_bc"] = DEFAULT_VALUE;        
    intMap["treat_outflow_bc"] = DEFAULT_VALUE;
    intMap["compute_velocity_magnitude"] = DEFAULT_VALUE;  
    intMap["float_to_rgb"] = DEFAULT_VALUE; 

    // populate data
    for (std::vector<std::string>::iterator i = lines.begin(); i != end; i++) {
        std::map<std::string, int>::iterator doubleIt;
        std::stringstream stream (*i);
        std::string key;
        int value;

        stream >> key >> value;
        if ((doubleIt = intMap.find(key)) != intMap.end()) {
            doubleIt->second = value;
        }

    }
    
    for (std::map<std::string, int>::iterator it = intMap.begin();
         it != intMap.end();
         ++it) {
        if (it->second == DEFAULT_VALUE) {
            PrintErrorMessage(it->first);
            exit(EXIT_FAILURE);
        }    
    }

    threads_distr.stream_device = intMap["stream_device"];
    threads_distr.update_density_field_device = intMap["update_density_field_device"];
    threads_distr.update_velocity_field_device = intMap["update_velocity_field_device"];
    threads_distr.update_population_field_device = intMap["update_population_field_device"];

    threads_distr.treat_non_slip_bc = intMap["treat_non_slip_bc"];
    threads_distr.treat_slip_bc = intMap["treat_slip_bc"];
    threads_distr.treat_inflow_bc = intMap["treat_inflow_bc"];
    threads_distr.treat_outflow_bc = intMap["treat_outflow_bc"];

    threads_distr.compute_velocity_magnitude = intMap["compute_velocity_magnitude"];
    threads_distr.float_to_rgb = intMap["float_to_rgb"];
}

int ComputeNumBlocks(const int num_threads, const int num_elements) {
    return (num_elements + num_threads) / num_threads;
}

void ComputeBlcoksDistr(struct CudaResourceDistr &blocks,
                        const struct CudaResourceDistr &threads,
                        const struct SimulationParametes &parameters,
                        const struct BoundaryConditions *boudnaries) {

    blocks.stream_device = ComputeNumBlocks(threads.stream_device,
                                            parameters.num_lattices);

    blocks.update_density_field_device = ComputeNumBlocks(threads.update_density_field_device,
                                                          parameters.num_lattices);

    blocks.update_velocity_field_device = ComputeNumBlocks(threads.update_velocity_field_device,
                                                           parameters.num_lattices);

    blocks.update_population_field_device = ComputeNumBlocks(threads.update_population_field_device,
                                                             parameters.num_lattices);

    blocks.treat_non_slip_bc = ComputeNumBlocks(threads.treat_non_slip_bc,
                                                boudnaries->num_wall_elements);

    blocks.treat_slip_bc = ComputeNumBlocks(threads.treat_slip_bc,
                                            boudnaries->num_moving_wall_elements);

    blocks.treat_inflow_bc = ComputeNumBlocks(threads.treat_inflow_bc,
                                              boudnaries->num_inflow_elements);

    blocks.treat_outflow_bc = ComputeNumBlocks(threads.treat_outflow_bc,
                                               boudnaries->num_outflow_elements);

    blocks.compute_velocity_magnitude = ComputeNumBlocks(threads.compute_velocity_magnitude,
                                                         parameters.num_lattices);

    blocks.float_to_rgb = ComputeNumBlocks(threads.float_to_rgb,
                                           parameters.num_lattices);
}

void DrawCircle(int x,
                int y,
                int marker,
                int* flag_field,
                const struct SimulationParametes &parameters) {
    int width = parameters.width;
    int radius = parameters.brush_size;

    int center[] = {x, y};
    for (int j = (center[1] - radius); j < (center[1] + radius); ++j) {
        for (int i = (center[0] - radius); i < (center[0] + radius); ++i) {
            real delta_x = real(center[0] - i);
            real delta_y = real(center[1] - j);
            real distance = sqrt(delta_x * delta_x + delta_y * delta_y);
            if (real(radius) > distance) {
                int index = GetIndex(i, j, width);
                flag_field[index] = marker;
            }
        }
    }
}
