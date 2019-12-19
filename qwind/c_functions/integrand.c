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

//size_t N_R;
//size_t N_DISK;
//size_t N_Z;
size_t n_eval = 50;
//double RG;
//double ASTAR = 0;
//double ISCO = 6;
//double EPSABS = 0;
//double EPSREL = 1e-4;
//double R_MIN = 6.;
//double R_MAX = 1600.;
//
//double *GRID_R_RANGE;
//double *GRID_Z_RANGE;
//double *DENSITY_GRID;
//
//double *GRID_DISK_RANGE;
//double *MDOT_GRID;
//double *UV_FRACTION_GRID;


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
    r_arg = get_arg(r, PARAMS.grid_r_range, PARAMS.n_r);
    z_arg = get_arg(z, PARAMS.grid_z_range, PARAMS.n_z);
    r_d_arg = get_arg(r_d, PARAMS.grid_r_range, PARAMS.n_r);
    z_d_arg = get_arg(0., PARAMS.grid_z_range, PARAMS.n_z);
    dr = abs(r_arg - r_d_arg);
    dz = abs(z_arg - z_d_arg);
    length = fmax(dr, dz) + 1;
    size_t *results = malloc(2*length * sizeof(size_t));
    drawline(r_d_arg, z_d_arg, r_arg, z_arg, results, length);
    position = results[1] * PARAMS.n_z + results[1 + length];
    for (int i=0; i<length; i++)
    {
        position = results[i] * PARAMS.n_z + results[i + length];
        tau += PARAMS.density_grid[position];
    }
    tau = tau / length * line_length;
    free(results);
    return tau;
}
double integrate_z_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    int r_arg;
    double delta, tau_uv, mdot, uv_fraction;
    //density_grid = params.density_grid
    r_arg = get_arg(PARAMS.r_d, PARAMS.grid_disk_range, PARAMS.n_disk); 
    mdot = PARAMS.mdot_grid[r_arg];
    uv_fraction = PARAMS.uv_fraction_grid[r_arg];
    tau_uv = tau_uv_disk_blob(PARAMS.r_d, phi_d, PARAMS.r, PARAMS.z) * SIGMA_T * PARAMS.R_g;
    delta = pow(PARAMS.r,2.) + pow(PARAMS.z,2.) + pow(PARAMS.r_d,2.) - 2 * PARAMS.r * PARAMS.r_d * cos(phi_d);
    integrand_value = uv_fraction * mdot * exp(-tau_uv) / pow(delta,2.);
    return integrand_value;
}
double integrate_z_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS.r_d = r_d;
    gsl_integration_cquad(gsl_integrate_z_phi_d, 0., M_PI, 0, PARAMS.epsrel, w, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_phi_d, 0., M_PI, 0, EPSREL, 1000, 1,w, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w);
    //double rel = ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, PARAMS.astar, PARAMS.isco);
    result = result * rel / pow(r_d, 2.);
    return result;
}
double integrate_z(struct parameters params)
{
    PARAMS=params;
    double result, error;
    printf("%e\n", PARAMS.epsrel);
    gsl_integration_cquad(gsl_integrate_z_r_d, PARAMS.r_min, PARAMS.r_max, 0, PARAMS.epsrel, w2, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, 1000, 1, w2, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, &result,  &n_eval, w2);
    result = 2. * pow(PARAMS.z,2.) * result;
    return result;
}
///////////////////////

double integrate_r_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    double cos_gamma;
    double delta, tau_uv, mdot, uv_fraction;
    int r_arg;
    r_arg = get_arg(PARAMS.r_d, PARAMS.grid_disk_range, PARAMS.n_disk); 
    mdot = PARAMS.mdot_grid[r_arg];
    uv_fraction = PARAMS.uv_fraction_grid[r_arg];
    cos_gamma = PARAMS.r - PARAMS.r_d * cos(phi_d);
    tau_uv = tau_uv_disk_blob(PARAMS.r_d, phi_d, PARAMS.r, PARAMS.z) * SIGMA_T * PARAMS.R_g;
    //printf("den: %e\n tau_uv: %e\n", density_grid[0], tau_uv);
    delta = pow(PARAMS.r,2.) + pow(PARAMS.z,2.) + pow(PARAMS.r_d,2.) - 2 * PARAMS.r * PARAMS.r_d * cos(phi_d);
    integrand_value = mdot * uv_fraction * cos_gamma * exp(-tau_uv) / pow(delta,2.);
    return integrand_value;
}
double integrate_r_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS.r_d = r_d;
    gsl_integration_cquad(gsl_integrate_r_phi_d, 0., M_PI, 0, PARAMS.epsrel, w3, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_r_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_phi_d, 0., M_PI, 0, EPSREL, 1000, 1, w3, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w3);
    //double rel =  ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, PARAMS.astar, PARAMS.isco);
    result = result * rel / pow(r_d, 2.);
    //printf("%e\n", result);
    return result;
}
double integrate_r(struct parameters params)
{
    PARAMS = params;
    printf("%e\n", PARAMS.r);
    printf("%e\n", PARAMS.z);
    printf("%e\n", PARAMS.epsrel);
    printf("%e\n", PARAMS.astar);
    printf("%e\n", PARAMS.density_grid[0]);
    double result, error;
    printf("epsrel\n");
    printf("%e\n", PARAMS.epsrel);
    printf("%e\n", params.epsrel);
    printf("%zu\n", PARAMS.n_disk);
    gsl_integration_cquad(gsl_integrate_r_r_d, PARAMS.r_min, PARAMS.r_max, 0, PARAMS.epsrel, w4, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_r_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_r_d, 6, 1600., 0, EPSREL, 1000, 1, w4, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_r_d, 6, 1600, 0, EPSREL, &result,  &n_eval, w4);
    result = 2. * PARAMS.z * result;
    return result;
}


/* NO TAU UV section */
double integrate_notau_z_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    int r_arg;
    double delta, mdot, uv_fraction;
    r_arg = get_arg(PARAMS.r_d, PARAMS.grid_disk_range, PARAMS.n_disk); 
    mdot = PARAMS.mdot_grid[r_arg];
    uv_fraction = PARAMS.uv_fraction_grid[r_arg];
    delta = pow(PARAMS.r,2.) + pow(PARAMS.z,2.) + pow(PARAMS.r_d,2.) - 2 * PARAMS.r * PARAMS.r_d * cos(phi_d);
    integrand_value = uv_fraction * mdot / pow(delta,2.);
    return integrand_value;
}
double integrate_notau_z_r_d(double r_d, void *params)
{
    double result, error;
    PARAMS.r_d = r_d;
    gsl_integration_cquad(gsl_integrate_notau_z_phi_d, 0., M_PI, 0, PARAMS.epsrel, w, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_phi_d, 0., M_PI, 0, EPSREL, 1000, 1,w, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w);
    //double rel =  ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, PARAMS.astar, PARAMS.isco);
    result = result *  rel / pow(r_d, 2.);
    return result;
}
double integrate_notau_z(struct parameters params)
{
    PARAMS = params;
    double result, error;
    gsl_integration_cquad(gsl_integrate_notau_z_r_d,
                          PARAMS.r_min,
                          PARAMS.r_max,
                          0,
                          PARAMS.epsrel,
                          w2, &result, &error, &n_eval);
    //gsl_integration_qng(gsl_integrate_z_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, 1000, 1, w2, &result, &error);
    //gsl_integration_romberg(gsl_integrate_z_r_d, 6, 1600., 0, EPSREL, &result,  &n_eval, w2);
    result = 2. * pow(PARAMS.z,2.) * result;
    return result;
}
///////////////////////

double integrate_notau_r_phi_d(double phi_d, void *params)
{
    // read all data
    //parameters params_gsl = *(parameters *)params;
    double integrand_value = 0;
    double cos_gamma;
    double delta, mdot, uv_fraction;
    int r_arg;
    r_arg = get_arg(PARAMS.r_d, PARAMS.grid_r_range, PARAMS.n_disk); 
    mdot = PARAMS.mdot_grid[r_arg];
    uv_fraction = PARAMS.uv_fraction_grid[r_arg];
    cos_gamma = PARAMS.r - PARAMS.r_d * cos(phi_d);
    delta = pow(PARAMS.r,2.) + pow(PARAMS.z,2.) + pow(PARAMS.r_d,2.) - 2 * PARAMS.r * PARAMS.r_d * cos(phi_d);
    integrand_value = mdot * uv_fraction * cos_gamma / pow(delta,2.);
    return integrand_value;
}
double integrate_notau_r_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS.r_d = r_d;
    gsl_integration_cquad(gsl_integrate_notau_r_phi_d,
                          0.,
                          M_PI,
                          0,
                          PARAMS.epsrel,
                          w3,
                          &result,
                          &error,
                          &n_eval);
    //gsl_integration_qng(gsl_integrate_r_phi_d, 0., M_PI, 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_phi_d, 0., M_PI, 0, EPSREL, 1000, 1, w3, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_phi_d, 0, M_PI, 0, EPSREL, &result,  &n_eval, w3);
    //double rel =  ( 1. - sqrt(6./r_d));
    double rel = nt_rel_factors(r_d, PARAMS.astar, PARAMS.isco);
    result = result *  rel / pow(r_d, 2.);
    //printf("%e\n", result);
    return result;
}
double integrate_notau_r(struct parameters params)
{
    //PARAMS.r = r;
    //PARAMS.z = z;
    PARAMS = params;
    double result, error;
    gsl_integration_cquad(gsl_integrate_notau_r_r_d,
                          PARAMS.r_min, PARAMS.r_max,
                          0,
                          PARAMS.epsrel,
                          w4,
                          &result,
                          &error,
                          &n_eval);
    //gsl_integration_qng(gsl_integrate_r_r_d, 6., 1600., 0, 1e-1, &result, &error, &n_eval);
    //gsl_integration_qag(gsl_integrate_r_r_d, 6, 1600., 0, EPSREL, 1000, 1, w4, &result, &error);
    //gsl_integration_romberg(gsl_integrate_r_r_d, 6, 1600, 0, EPSREL, &result,  &n_eval, w4);
    result = 2. * params.z * result;
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

int main()
{
    double *grid_r_range = malloc(10 * sizeof(double));
    double *grid_z_range = malloc(10 * sizeof(double));
    double *density_grid = malloc(10 * 10 * sizeof(double));
    double *mdot_grid= malloc(10 * sizeof(double));
    double *uv_fraction_grid = malloc(10 * sizeof(double));
    double *grid_disk_range = malloc( 10 * sizeof(double));
    ////parameters param_int;
    //////double *density_grid = malloc(100 * sizeof(double));
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
    parameters param_ex;
    param_ex.grid_r_range = grid_r_range;
    param_ex.grid_z_range = grid_z_range;
    param_ex.grid_disk_range = grid_disk_range;
    param_ex.density_grid = density_grid;
    param_ex.R_g = 1e14;
    param_ex.n_r = 10;
    param_ex.n_z = 10;
    param_ex.n_disk= 10;
    param_ex.mdot_grid = mdot_grid;
    param_ex.uv_fraction_grid = uv_fraction_grid;
    param_ex.epsrel = 1e-4;
    param_ex.r = 100;
    param_ex.z = 50;
    param_ex.astar =0.;
    param_ex.isco = 6.;
    param_ex.r_min = 6.;
    param_ex.r_max = 1600.;

    initialize_integrators();
    double result;
    result = integrate_z(param_ex);
    printf("%e\n", result);
    result = integrate_r(param_ex);
    printf("%e\n", result);
    result = integrate_notau_z(param_ex);
    printf("%e\n", result);
    result = integrate_notau_r(param_ex);
    printf("%e\n", result);
    //initialize_arrays(grid_r_range, grid_z_range, density_grid, mdot_grid, uv_fraction_grid, grid_disk_range, 10, 10, 10, 1e14, 1e-4);
    ////param_int.r = 100.;
    ////param_int.z = 50.;
    ////////param_int.density_grid = density_grid;
    ////param_int.Rg = 1e14;
    ////double result;
    ////double x[2];
    ////x[0] = 20;
    ////x[1] = 0;
    ////result = integrand_z(2, x, &param_int);
    ////printf("%e\n", result);
    //double result, result2;
    //result = integrate_z(50, 100);
    //result2 = integrate_r(50, 100);
    //printf("%e\n",result2);
    //printf("%e\n",result);
    //printf("no tau:\n");
    //result = integrate_notau_z(50, 100);
    //result2 = integrate_notau_r(50, 100);
    //printf("%e\n",result2);
    //printf("%e\n",result);
    //printf("now i modify mdot grid\n");
    // for (int i=0; i<10; i++)
    //{
    //    mdot_grid[i] = 0.5;
    //}
    ////initialize_arrays(grid_r_range, grid_z_range, density_grid, mdot_grid, uv_fraction_grid, grid_disk_range, 10, 10, 10, 1e14, 1e-4);
    //result = integrate_notau_z(50, 100);
    //result2 = integrate_notau_r(50, 100);
    //printf("%e\n",result2);
    //printf("%e\n",result);
    //double tau;
    //tau = tau_uv_disk_blob(10, 0, 100, 500);
//ta//u = tau * 1e14 * 1e-25;
    //printf("\ntau2: %f \n",tau);

    return 0;
}





