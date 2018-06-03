#include <vector>
#include <stdio.h>
#include "headers/parameters.h"
#include "headers/scanning.h"
#include "headers/boundary_conditions.h"
#include "headers/domain.h"
#include "headers/helper.h"

void ScanFlagField(int *flag_field,
                   DomainHandler &domain_handler,
                   BoundaryConditionsHandler &bc_handler,
                   const struct SimulationParametes &parameters,
                   const struct Constants &constants,
                   const struct BoundaryInfo &boundary_info) {
    int num_directions = parameters.discretization;
    int width = parameters.width;

    std::vector<int> FluidIndices;
    std::vector<int> SolidIndices;
    std::vector<struct WallBC> wall_elements;
    std::vector<struct MovingWallBC> mov_wall_elements;
    std::vector<struct InflowBC> inflow_elements;
    std::vector<struct OutflowBC> outflow_elements;

    for (int j = 0; j < parameters.height; ++j) {
        for (int i = 0; i < parameters.width; ++i) {
            int scalar_self_index = GetIndex(i, j, width);

            if (flag_field[scalar_self_index] == FLUID) {
                FluidIndices.push_back(scalar_self_index);

                for (int component = 0; component < num_directions; ++component) {
                    int ii = coords[component];
                    int jj = coords[component + num_directions];

                    int scalar_neighbour_index = GetIndex(i + ii, j + jj, width);
                    int shift = component * parameters.num_lattices;
                    
                    int neighbour_flag = flag_field[scalar_neighbour_index];

                    if (neighbour_flag != FLUID) {

                        switch (neighbour_flag) {
                            case WALL: 
                                struct WallBC wall;
                                PrecomputeWallBC(component,
                                                 scalar_self_index,
                                                 wall,
                                                 parameters);

                                wall_elements.push_back(wall);
                                break;
                            case MOVING_WALL: 
                                struct MovingWallBC moving_wall;
                                PrecomputeMovingWallBC(component,
                                                       scalar_self_index,
                                                       moving_wall,
                                                       parameters,
                                                       constants,
                                                       boundary_info);

                                mov_wall_elements.push_back(moving_wall);
                                break;
                            case INFLOW: 
                                struct InflowBC inlet_lattice;
                                
                                PrecomputeInflowBC(component,
                                                   scalar_self_index,
                                                   inlet_lattice,
                                                   parameters,
                                                   constants,
                                                   boundary_info);

                                inflow_elements.push_back(inlet_lattice); 
                                break;
                            case OUTFLOW: 
                                struct OutflowBC outlet_lattice;
                                PrecomputeOutflowBC(component,
                                                    scalar_self_index,
                                                    outlet_lattice,
                                                    parameters);

                                outflow_elements.push_back(outlet_lattice);
                                break;
                            default:
                                break;
                        }

                    }
                }
            }
            else {
                SolidIndices.push_back(scalar_self_index); 
            }
        }
    }

    bc_handler.SetNonSlipBC(wall_elements);
    bc_handler.SetSlipBC(mov_wall_elements);
    bc_handler.SetInflowBC(inflow_elements);
    bc_handler.SetOutflowBC(outflow_elements);
    domain_handler.AllocateFluidElementIndices(FluidIndices);
    domain_handler.AllocateSolidElementIndices(SolidIndices);
}
