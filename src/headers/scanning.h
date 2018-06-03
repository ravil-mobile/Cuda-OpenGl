#include "parameters.h"
#include "domain.h"
#include "boundary_conditions.h"

#ifndef SEQUENTIAL_LBM_SRC_HEADERS_SCANNING_H_
#define SEQUENTIAL_LBM_SRC_HEADERS_SCANNING_H_

void ScanFlagField(int *flag_field,
                   DomainHandler &domain_handler,
                   BoundaryConditionsHandler &bc_handler,
                   const struct SimulationParametes &parameters,
                   const struct Constants &constants,
                   const struct BoundaryInfo &boundary_info);

#endif  // SEQUENTIAL_LBM_SRC_HEADERS_SCANNING_H_
