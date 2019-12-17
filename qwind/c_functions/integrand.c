#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "grid_utils.c"
#define SIGMA_T 6.6524587158e-25
//#define N_R 1000
//#define N_Z 1001

size_t N_R;
size_t N_DISK;
size_t N_Z;

double *GRID_R_RANGE;
double *GRID_Z_RANGE;
double *DENSITY_GRID;

double *GRID_DISK_RANGE;
double *MDOT_GRID;
double *UV_FRACTION_GRID;

void initialize_arrays(double *grid_r_range, double *grid_z_range, double *density_grid, double *mdot_grid, double *uv_fraction_grid, double *grid_disk_range, size_t n_r, size_t n_z, size_t n_disk)
{
    N_R = n_r;
    N_Z = n_z;
    N_DISK = n_disk;
    GRID_R_RANGE = malloc(n_r * sizeof(double));
    for (int i=0; i < n_r; i++)
    {
        GRID_R_RANGE[i] = grid_r_range[i];
    }
    GRID_Z_RANGE = malloc(n_z * sizeof(double));
    for (int i=0; i < n_z; i++)
    {
        GRID_Z_RANGE[i] = grid_z_range[i];
    }
    DENSITY_GRID = malloc(n_z * n_r * sizeof(double));
    for (int i=0; i < n_r * n_z; i++)
    {
        DENSITY_GRID[i] = density_grid[i];
    }

    MDOT_GRID = malloc(n_disk * sizeof(double));
    for (int i=0; i < n_disk; i++)
    {
        MDOT_GRID[i] = mdot_grid[i];
    }
    UV_FRACTION_GRID = malloc(n_disk * sizeof(double));
    for (int i=0; i < n_disk; i++)
    {
        UV_FRACTION_GRID[i] = uv_fraction_grid[i];
    }
    GRID_DISK_RANGE = malloc(n_disk * sizeof(double));
    for (int i=0; i < n_disk; i++)
    {
        GRID_DISK_RANGE[i] = GRID_DISK_RANGE[i];
    }
}

typedef struct parameters{
    double r;
    double z;
    double Rg;
} parameters;

double integrand_r(int n, double *x, void *user_data)
{
    // read all data
    parameters params = *(parameters *)user_data;
    double integrand_value = 0;
    double r_d, phi_d;
    double r, z;
    double delta, tau_uv, mdot, uv_fraction;
    //double *density_grid;
    double Rg = params.Rg;
    double cos_gamma;
    r_d = x[0];
    phi_d = x[1];
    r = params.r;
    z = params.z;
    //density_grid = params.density_grid;
    cos_gamma = r - r_d * cos(phi_d);
    tau_uv = tau_uv_disk_blob(r_d, phi_d, r, z, DENSITY_GRID, GRID_R_RANGE, GRID_Z_RANGE, N_R, N_Z) * SIGMA_T * Rg;
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = ( 1. - sqrt(6./r_d)) / pow(r_d, 2.) / pow(delta,2.) * exp(-tau_uv) * cos_gamma;
    return integrand_value;
}

double integrand_z(int n, double *x, void *user_data)
{
    // read all data
    parameters params = *(parameters *)user_data;
    double integrand_value = 0;
    double r_d, phi_d;
    double r, z;
    double delta, tau_uv, mdot, uv_fraction;
    //double *density_grid;
    double Rg = params.Rg;
    r_d = x[0];
    phi_d = x[1];
    r = params.r;
    z = params.z;
    //density_grid = params.density_grid;
    tau_uv = tau_uv_disk_blob(r_d, phi_d, r, z, DENSITY_GRID, GRID_R_RANGE, GRID_Z_RANGE, N_R, N_Z) * SIGMA_T * Rg;
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = ( 1. - sqrt(6./r_d)) / pow(r_d, 2.) / pow(delta,2.) * exp(-tau_uv);
    return integrand_value;
}


int main()
{
    double *grid_r_range = malloc(10 * sizeof(double));
    double *grid_z_range = malloc(10 * sizeof(double));
    double *density_grid = malloc(10 * 10 * sizeof(double));
    double *mdot_grid= malloc(10 * sizeof(double));
    double *uv_fraction_grid = malloc(10 * sizeof(double));
    double *grid_disk_range = malloc( 10 * sizeof(double));
    parameters param_int;
    ////double *density_grid = malloc(100 * sizeof(double));
    for (int i=0; i<100; i++)
    {
        density_grid[i] = 2e8;
    }
    for (int i=0; i<10; i++)
    {
        grid_r_range[i] = 10*i;
        grid_z_range[i] = 10*i;
        mdot_grid[i] = 0.5;
        uv_fraction_grid[i] = 0.9;
        grid_disk_range[i] = 10*i;
    }

    initialize_arrays(grid_r_range, grid_z_range, density_grid, mdot_grid, uv_fraction_grid, grid_disk_range, 10, 10, 10);
    param_int.r = 100.;
    param_int.z = 50.;
    ////param_int.density_grid = density_grid;
    param_int.Rg = 1e14;
    double result;
    double x[2];
    x[0] = 20;
    x[1] = 0;
    result = integrand_z(2, x, &param_int);
    printf("%e\n", result);
    return 0;
}





