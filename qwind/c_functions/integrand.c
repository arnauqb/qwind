#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "grid_utils.h"
#include "disk.h"
#include "integrand.h"
#include <gsl/gsl_integration.h>
#define SIGMA_T 6.6524587158e-25
//#define N_R 1000
//#define N_Z 1001

size_t N_R;
size_t N_DISK;
size_t N_Z;
size_t n_eval;
double RG;
double ASTAR = 0;
double ISCO = 6;
double EPSABS = 0;
double EPSREL = 1e-4;
double R_MIN = 6.;
double R_MAX = 1600.;

double *GRID_R_RANGE;
double *GRID_Z_RANGE;
double *DENSITY_GRID;

double *GRID_DISK_RANGE;
double *MDOT_GRID;
double *UV_FRACTION_GRID;

typedef struct parameters{
    double r;
    double z;
    double r_d_gsl;
} parameters;

parameters PARAMS;

//gsl_integration_workspace *w = NULL;//
//gsl_integration_workspace *w2 = NULL;//
//gsl_integration_workspace *w3 = NULL;//
//gsl_integration_workspace *w4 = NULL;//

gsl_integration_cquad_workspace *w = NULL;//
gsl_integration_cquad_workspace *w2 = NULL;//
gsl_integration_cquad_workspace *w3 = NULL;//
gsl_integration_cquad_workspace *w4 = NULL;//

gsl_function *gsl_integrate_z_phi_d = NULL; 
gsl_function *gsl_integrate_z_r_d = NULL; 
gsl_function *gsl_integrate_r_phi_d = NULL; 
gsl_function *gsl_integrate_r_r_d = NULL; 


gsl_function *gsl_integrate_notau_z_phi_d = NULL; 
gsl_function *gsl_integrate_notau_z_r_d = NULL; 
gsl_function *gsl_integrate_notau_r_phi_d = NULL; 
gsl_function *gsl_integrate_notau_r_r_d = NULL; 

// functions

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
//parameters params;
//gsl_function gsl_integrand_z_phi_d;
//gsl_integrand_z_phi_d.function = &integrand_z_phi_d;
//gsl_integrand_z_phi_d.params = &params;
//
double tau_uv_disk_blob(double r_d, double phi_d, double r, double z)
{
    double line_length;
    size_t r_arg, z_arg, r_d_arg, z_d_arg;
    double dr, dz;
    double tau = 0;
    size_t length, position; 
    line_length = sqrt(pow(r,2.) + pow(r_d,2.) + pow(z,2.) - 2 * r * r_d * cos(phi_d));
    r_arg = get_arg(r, GRID_R_RANGE, N_R);
    z_arg = get_arg(z, GRID_Z_RANGE, N_Z);
    r_d_arg = get_arg(r_d, GRID_R_RANGE, N_R);
    z_d_arg = get_arg(0., GRID_Z_RANGE, N_Z);
    dr = abs(r_arg - r_d_arg);
    dz = abs(z_arg - z_d_arg);
    length = fmax(dr, dz) + 1;
    size_t *results = malloc(2*length * sizeof(size_t));
    drawline(r_d_arg, z_d_arg, r_arg, z_arg, results, length);
    position = results[1] * N_Z + results[1 + length];
    for (int i=0; i<length; i++)
    {
        position = results[i] * N_Z + results[i + length];
        tau += DENSITY_GRID[position];
    }
    tau = (tau / (double)(length) * line_length);
    free(results);
    return tau;
}
double integrate_z_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    double r_d;
    double r, z;
    int r_arg;
    double delta, tau_uv, mdot, uv_fraction;
    r = PARAMS.r;
    z = PARAMS.z;
    r_d = PARAMS.r_d_gsl;
    //density_grid = params.density_grid
    r_arg = get_arg(r_d, GRID_DISK_RANGE, N_DISK); 
    mdot = MDOT_GRID[r_arg];
    uv_fraction = UV_FRACTION_GRID[r_arg];
    tau_uv = tau_uv_disk_blob(r_d, phi_d, r, z) * SIGMA_T * RG;
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = uv_fraction * mdot * exp(-tau_uv) / pow(delta,2.);
    return integrand_value;
}
double integrate_z_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS.r_d_gsl = r_d;
    gsl_integration_cquad(gsl_integrate_z_phi_d, 0., M_PI, 0, EPSREL, w, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_phi_d, 0., M_PI, 0, EPSREL, 1000, 1,w, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w);
    //double rel = ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, ASTAR, ISCO);
    result = result * rel / pow(r_d, 2.);
    return result;
}
double integrate_z(double r, double z)
{
    PARAMS.r = r;
    PARAMS.z = z;
    double result, error;
    gsl_integration_cquad(gsl_integrate_z_r_d, R_MIN, R_MAX, 0, EPSREL, w2, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, 1000, 1, w2, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, &result,  &n_eval, w2);
    result = 2. * pow(z,2.) * result;
    return result;
}
///////////////////////

double integrate_r_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    double r_d;
    double r, z, cos_gamma;
    double delta, tau_uv, mdot, uv_fraction;
    int r_arg;
    r = PARAMS.r;
    z = PARAMS.z;
    r_d = PARAMS.r_d_gsl;
    r_arg = get_arg(r_d, GRID_DISK_RANGE, N_DISK); 
    mdot = MDOT_GRID[r_arg];
    uv_fraction = UV_FRACTION_GRID[r_arg];
    cos_gamma = r - r_d * cos(phi_d);
    tau_uv = tau_uv_disk_blob(r_d, phi_d, r, z) * SIGMA_T * RG;
    //printf("den: %e\n tau_uv: %e\n", DENSITY_GRID[0], tau_uv);
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = mdot * uv_fraction * cos_gamma * exp(-tau_uv) / pow(delta,2.);
    return integrand_value;
}
double integrate_r_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS.r_d_gsl = r_d;
    gsl_integration_cquad(gsl_integrate_r_phi_d, 0., M_PI, 0, EPSREL, w3, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_r_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_phi_d, 0., M_PI, 0, EPSREL, 1000, 1, w3, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w3);
    //double rel =  ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, ASTAR, ISCO);
    result = result * rel / pow(r_d, 2.);
    //printf("%e\n", result);
    return result;
}
double integrate_r(double r, double z)
{
    PARAMS.r = r;
    PARAMS.z = z;
    double result, error;
    gsl_integration_cquad(gsl_integrate_r_r_d, R_MIN, R_MAX, 0, EPSREL, w4, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_r_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_r_d, 6, 1600., 0, EPSREL, 1000, 1, w4, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_r_d, 6, 1600, 0, EPSREL, &result,  &n_eval, w4);
    result = 2. * z * result;
    return result;
}


/* NO TAU UV section */
double integrate_notau_z_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    double r_d;
    double r, z;
    int r_arg;
    double delta, mdot, uv_fraction;
    r = PARAMS.r;
    z = PARAMS.z;
    r_d = PARAMS.r_d_gsl;
    r_arg = get_arg(r_d, GRID_DISK_RANGE, N_DISK); 
    mdot = MDOT_GRID[r_arg];
    uv_fraction = UV_FRACTION_GRID[r_arg];
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = uv_fraction * mdot / pow(delta,2.);
    return integrand_value;
}
double integrate_notau_z_r_d(double r_d, void *params)
{
    double result, error;
    PARAMS.r_d_gsl = r_d;
    gsl_integration_cquad(gsl_integrate_notau_z_phi_d, 0., M_PI, 0, EPSREL, w, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_phi_d, 0., M_PI, 0, EPSREL, 1000, 1,w, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w);
    //double rel =  ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, ASTAR, ISCO);
    result = result *  rel / pow(r_d, 2.);
    return result;
}
double integrate_notau_z(double r, double z)
{
    PARAMS.r = r;
    PARAMS.z = z;
    double result, error;
    gsl_integration_cquad(gsl_integrate_notau_z_r_d, R_MIN, R_MAX, 0, EPSREL, w2, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, 1000, 1, w2, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, &result,  &n_eval, w2);
    result = 2. * pow(z,2.) * result;
    return result;
}
///////////////////////

double integrate_notau_r_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    double r_d;
    double r, z, cos_gamma;
    double delta, mdot, uv_fraction;
    int r_arg;
    r = PARAMS.r;
    z = PARAMS.z;
    r_d = PARAMS.r_d_gsl;
    r_arg = get_arg(r_d, GRID_DISK_RANGE, N_DISK); 
    mdot = MDOT_GRID[r_arg];
    uv_fraction = UV_FRACTION_GRID[r_arg];
    cos_gamma = r - r_d * cos(phi_d);
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = mdot * uv_fraction * cos_gamma / pow(delta,2.);
    return integrand_value;
}
double integrate_notau_r_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS.r_d_gsl = r_d;
    gsl_integration_cquad(gsl_integrate_notau_r_phi_d, 0., M_PI, 0, EPSREL, w3, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_r_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_phi_d, 0., M_PI, 0, EPSREL, 1000, 1, w3, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w3);
    //double rel =  ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, ASTAR, ISCO);
    result = result *  rel / pow(r_d, 2.);
    //printf("%e\n", result);
    return result;
}
double integrate_notau_r(double r, double z)
{
    PARAMS.r = r;
    PARAMS.z = z;
    double result, error;
    gsl_integration_cquad(gsl_integrate_notau_r_r_d, R_MIN, R_MAX, 0, EPSREL, w4, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_r_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_r_d, 6, 1600., 0, EPSREL, 1000, 1, w4, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_r_d, 6, 1600, 0, EPSREL, &result,  &n_eval, w4);
    result = 2. * z * result;
    return result;
}


void initialize_integrators()
{
    n_eval=50;
    gsl_integrate_z_phi_d = malloc(sizeof(gsl_function));
    gsl_integrate_z_phi_d->function = &integrate_z_phi_d;
    gsl_integrate_z_phi_d->params = NULL;
    w = gsl_integration_cquad_workspace_alloc(n_eval);
    //w = gsl_integration_workspace_alloc(1000); 
    //w = gsl_integration_romberg_alloc(n_eval);

    gsl_integrate_z_r_d = malloc(sizeof(gsl_function));
    gsl_integrate_z_r_d->function = &integrate_z_r_d;
    gsl_integrate_z_r_d->params = NULL;
    w2 = gsl_integration_cquad_workspace_alloc(n_eval);
    //w2 = gsl_integration_workspace_alloc(1000); 
    //w2 = gsl_integration_romberg_alloc(n_eval);

    gsl_integrate_r_phi_d = malloc(sizeof(gsl_function));
    gsl_integrate_r_phi_d->function = &integrate_r_phi_d;
    gsl_integrate_r_phi_d->params = NULL;
    w3 = gsl_integration_cquad_workspace_alloc(n_eval);
    //w3 = gsl_integration_workspace_alloc(1000); 
    //w3 = gsl_integration_romberg_alloc(n_eval);

    gsl_integrate_r_r_d = malloc(sizeof(gsl_function));
    gsl_integrate_r_r_d->function = &integrate_r_r_d;
    gsl_integrate_r_r_d->params = NULL;
    w4 = gsl_integration_cquad_workspace_alloc(n_eval);
    //w4 = gsl_integration_workspace_alloc(1000); 
    //w4 = gsl_integration_romberg_alloc(n_eval);

    gsl_integrate_notau_z_phi_d = malloc(sizeof(gsl_function));
    gsl_integrate_notau_z_phi_d->function = &integrate_notau_z_phi_d;
    gsl_integrate_notau_z_phi_d->params = NULL;

    gsl_integrate_notau_z_r_d = malloc(sizeof(gsl_function));
    gsl_integrate_notau_z_r_d->function = &integrate_notau_z_r_d;
    gsl_integrate_notau_z_r_d->params = NULL;

    gsl_integrate_notau_r_phi_d = malloc(sizeof(gsl_function));
    gsl_integrate_notau_r_phi_d->function = &integrate_notau_r_phi_d;
    gsl_integrate_notau_r_phi_d->params = NULL;

    gsl_integrate_notau_r_r_d = malloc(sizeof(gsl_function));
    gsl_integrate_notau_r_r_d->function = &integrate_notau_r_r_d;
    gsl_integrate_notau_r_r_d->params = NULL;

}

void initialize_arrays(double *grid_r_range, double *grid_z_range, double *density_grid, double *mdot_grid, double *uv_fraction_grid, double *grid_disk_range, size_t n_r, size_t n_z, size_t n_disk, double Rg, double epsrel)
{
    N_R = n_r;
    N_Z = n_z;
    N_DISK = n_disk;
    RG = Rg;
    EPSREL = epsrel;
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
    printf("%e\n", DENSITY_GRID[0]);

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
        GRID_DISK_RANGE[i] = grid_disk_range[i];
    }
    R_MIN = GRID_DISK_RANGE[0];
    R_MAX = GRID_DISK_RANGE[N_DISK-1];
}

double integrand_r(int n, double *x, void *user_data)
{
    // read all data
    parameters params = *(parameters *)user_data;
    double integrand_value = 0;
    double r_d, phi_d;
    double r, z;
    double delta, tau_uv;
    //double *density_grid;
    double cos_gamma;
    r_d = x[0];
    phi_d = x[1];
    r = params.r;
    z = params.z;
    //density_grid = params.density_grid;
    cos_gamma = r - r_d * cos(phi_d);
    tau_uv = tau_uv_disk_blob(r_d, phi_d, r, z) * SIGMA_T * RG;
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = nt_rel_factors(r_d, ASTAR, ISCO) / pow(r_d, 2.) / pow(delta,2.) * exp(-tau_uv) * cos_gamma;
    return integrand_value;
}

double integrand_z(int n, double *x, void *user_data)
{
    // read all data
    parameters params = *(parameters *)user_data;
    double integrand_value = 0;
    double r_d, phi_d;
    double r, z;
    double delta, tau_uv;
    //double *density_grid;
    r_d = x[0];
    phi_d = x[1];
    r = params.r;
    z = params.z;
    //density_grid = params.density_grid;
    tau_uv = tau_uv_disk_blob(r_d, phi_d, r, z) * SIGMA_T * RG;
    delta = pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d);
    integrand_value = nt_rel_factors(r_d, ASTAR,ISCO) / pow(r_d, 2.) / pow(delta,2.) * exp(-tau_uv);
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
    //parameters param_int;
    ////double *density_grid = malloc(100 * sizeof(double));
    for (int i=0; i<100; i++)
    {
        density_grid[i] = 2e8;
    }
    for (int i=0; i<10; i++)
    {
        grid_r_range[i] = 10*i;
        grid_z_range[i] = 10*i;
        mdot_grid[i] = 1.;
        uv_fraction_grid[i] = 1.;
        grid_disk_range[i] = 6 + 10*i;
    }

    initialize_arrays(grid_r_range, grid_z_range, density_grid, mdot_grid, uv_fraction_grid, grid_disk_range, 10, 10, 10, 1e14, 1e-4);
    initialize_integrators();
    //param_int.r = 100.;
    //param_int.z = 50.;
    //////param_int.density_grid = density_grid;
    //param_int.Rg = 1e14;
    //double result;
    //double x[2];
    //x[0] = 20;
    //x[1] = 0;
    //result = integrand_z(2, x, &param_int);
    //printf("%e\n", result);
    double result, result2;
    result = integrate_z(50, 100);
    result2 = integrate_r(50, 100);
    printf("%e\n",result2);
    printf("%e\n",result);
    printf("no tau:\n");
    result = integrate_notau_z(50, 100);
    result2 = integrate_notau_r(50, 100);
    printf("%e\n",result2);
    printf("%e\n",result);
    printf("now i modify mdot grid\n");
     for (int i=0; i<10; i++)
    {
        mdot_grid[i] = 0.5;
    }
    initialize_arrays(grid_r_range, grid_z_range, density_grid, mdot_grid, uv_fraction_grid, grid_disk_range, 10, 10, 10, 1e14, 1e-4);
    result = integrate_notau_z(50, 100);
    result2 = integrate_notau_r(50, 100);
    printf("%e\n",result2);
    printf("%e\n",result);
    double tau;
    tau = tau_uv_disk_blob(10, 0, 100, 500);
//tau = tau * 1e14 * 1e-25;
    printf("\ntau2: %f \n",tau);

    return 0;
}





