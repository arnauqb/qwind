#include <stdio.h>
double tau_uv_disk_blob(double, double, double, double);
double integrate_z_phi_d(double phi_d, void *params);
double integrate_z_r_d(double r_d , void *params);
double integrate_z(double r, double z);
double integrate_r_phi_d(double phi_d, void *params);
double integrate_r_r_d(double r_d , void *params);
double integrate_r(double r, double z);
double integrate_notau_z_phi_d(double phi_d, void *params);
double integrate_notau_z_r_d(double r_d, void *params);
double integrate_notau_z(double r, double z);
double integrate_notau_r_phi_d(double phi_d, void *params);
double integrate_notau_r_r_d(double r_d , void *params);
double integrate_notau_r(double r, double z);
void initialize_integrators();
void initialize_arrays(double *grid_r_range, double *grid_z_range, double *density_grid, double *mdot_grid, double *uv_fraction_grid, double *grid_disk_range, size_t n_r, size_t n_z, size_t n_disk, double Rg, double epsrel);
double integrand_r(int n, double *x, void *user_data);
double integrand_z(int n, double *x, void *user_data);
