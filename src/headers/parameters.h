#ifndef SEQUENTIAL_LBM_SRC_HEADERS_PARAMETERS_H_
#define SEQUENTIAL_LBM_SRC_HEADERS_PARAMETERS_H_

typedef float real;

struct SimulationParametes {
    real simulation_time;
    int num_time_steps;
    int dimension;
    int discretization;
    real delta_x;
    real delta_t;
    real speed_of_sound;
    real viscosity;
    real tau;
    real relaxation;
    int width;
    int height;
    int num_lattices;
    int steps_per_report;
    real max_velocity_rendering;
    real min_velocity_rendering;
    int brush_size;
};
//extern struct SimulationParametes parameters;

struct Constants {
    real one;
    real two;
    real three;
};
//extern struct Constants constants;

struct BoundaryInfo {
    real wall_velocity_x;
    real wall_velocity_y;
    real velocity_inflow_x;
    real velocity_inflow_y;
    real density_outflow;
};
//extern struct BoundaryInfo boundary_info;

enum flags {FLUID, WALL, MOVING_WALL, INFLOW, OUTFLOW};
enum func_type{ONE, TWO};
enum thread_type {MASTER, SLAVE};

extern int coords[];
extern real weights[];
extern int inverse_indices[];

#endif  // SEQUENTIAL_LBM_SRC_HEADERS_PARAMETERS_H_

