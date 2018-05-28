#include "parameters.h"

#ifndef SEQUENTIAL_LBM_SRC_HEADERS_DOMAIN_H_
#define SEQUENTIAL_LBM_SRC_HEADERS_DOMAIN_H_

struct Domain {
    int *dev_flag_field; 
    real *dev_density;
    real *dev_velocity;
    real *dev_velocity_magnitude;
    real *dev_population; 
    real *dev_swap_buffer;
};


class DomainHandler {
public:
    DomainHandler();
    ~DomainHandler();
    
    void InitDomainOnDevice(SimulationParametes &parameters,
                            int *flag_field);
    
    void SwapPopulationFields();
    const Domain * GetDeviceData();
private:
    Domain dev_domain;
};


#endif  // SEQUENTIAL_LBM_SRC_HEADERS_DOMAIN_H_

