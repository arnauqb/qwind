#include <stdio.h>
typedef struct parameters{
    double r;
    double z;
    double r_d;
    double *grid_r_range;
    double *grid_z_range;
    size_t n_r;
    size_t n_z;
    double *density_grid;
    double *uv_fraction_grid;
    double *mdot_grid;
    double *grid_disk_range;
    size_t n_disk;
    double R_g;
    double astar;
    double isco;
    double r_min;
    double r_max;
    double epsabs;
    double epsrel;
} parameters;

void test(parameters *params);
double tau_uv_disk_blob(double, double, double, double);
double integrate_z_phi_d(double phi_d, void *params);
double integrate_z_r_d(double r_d , void *params);
double integrate_z(parameters*);
double integrate_r_phi_d(double phi_d, void *params);
double integrate_r_r_d(double r_d , void *params);
double integrate_r(parameters*);

double integrate_notau_z_phi_d(double phi_d, void *params);
double integrate_notau_z_r_d(double r_d, void *params);
double integrate_notau_z(parameters*);
double integrate_notau_r_phi_d(double phi_d, void *params);
double integrate_notau_r_r_d(double r_d , void *params);
double integrate_notau_r(parameters*);

double integrate_simplesed_z_phi_d(double phi_d, void *params);
double integrate_simplesed_z_r_d(double r_d, void *params);
double integrate_simplesed_z(parameters*);
double integrate_simplesed_r_phi_d(double phi_d, void *params);
double integrate_simplesed_r_r_d(double r_d , void *params);
double integrate_simplesed_r(parameters*);

void initialize_integrators();
void initialize_arrays(double *grid_r_range, double *grid_z_range, double *density_grid, double *mdot_grid, double *uv_fraction_grid, double *grid_disk_range, size_t n_r, size_t n_z, size_t n_disk, double Rg, double epsrel);
