#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_integration.h>
#include "disk.h"
#include "integrand.h"
#define SIGMA_T 6.6524587158e-25

size_t n_eval = 50;

gsl_integration_cquad_workspace *w = NULL;//
gsl_integration_cquad_workspace *w2 = NULL;//
gsl_integration_cquad_workspace *w3 = NULL;//
gsl_integration_cquad_workspace *w4 = NULL;//

gsl_function *gsl_integrate_simplesed_z_phi_d = NULL; 
gsl_function *gsl_integrate_simplesed_z_r_d = NULL; 
gsl_function *gsl_integrate_simplesed_r_phi_d = NULL; 
gsl_function *gsl_integrate_simplesed_r_r_d = NULL; 

static parameters *PARAMS = NULL;

// simple sed
double integrate_simplesed_z_phi_d(double phi_d, void *params)
{
    double integrand_value = 0;
    double delta;
    delta = pow(PARAMS->r,2.) + pow(PARAMS->z,2.) + pow(PARAMS->r_d,2.) - 2 * PARAMS->r * PARAMS->r_d * cos(phi_d);
    integrand_value = 1. / pow(delta,2.);
    return integrand_value;
}
double integrate_simplesed_z_r_d(double r_d, void *params)
{
    double result, error;
    PARAMS->r_d = r_d;
    gsl_integration_cquad(gsl_integrate_simplesed_z_phi_d, 0., M_PI, 0, PARAMS->epsrel, w, &result, &error, &n_eval);
    double rel = nt_rel_factors(r_d, PARAMS->astar, PARAMS->isco);
    result = result *  rel / pow(r_d, 2.);
    return result;
}
double integrate_simplesed_z(parameters *params)
{
    PARAMS = params;
    double result, error;
    gsl_integration_cquad(gsl_integrate_simplesed_z_r_d,
                          PARAMS->r_min,
                          PARAMS->r_max,
                          0,
                          PARAMS->epsrel,
                          w2, &result, &error, &n_eval);
    result = 2. * pow(PARAMS->z,2.) * result;
    return result;
}

double integrate_simplesed_r_phi_d(double phi_d, void *params)
{
    double integrand_value = 0;
    double cos_gamma;
    double delta;
    cos_gamma = PARAMS->r - PARAMS->r_d * cos(phi_d);
    delta = pow(PARAMS->r,2.) + pow(PARAMS->z,2.) + pow(PARAMS->r_d,2.) - 2 * PARAMS->r * PARAMS->r_d * cos(phi_d);
    integrand_value = cos_gamma / pow(delta,2.);
    return integrand_value;
}
double integrate_simplesed_r_r_d(double r_d , void *params)
{
    double result, error;
    PARAMS->r_d = r_d;
    gsl_integration_cquad(gsl_integrate_simplesed_r_phi_d,
                          0.,
                          M_PI,
                          0,
                          PARAMS->epsrel,
                          w3,
                          &result,
                          &error,
                          &n_eval);
    double rel = nt_rel_factors(r_d, PARAMS->astar, PARAMS->isco);
    result = result *  rel / pow(r_d, 2.);
    return result;
}
double integrate_simplesed_r(parameters *params)
{
    PARAMS = params;
    double result, error;
    gsl_integration_cquad(gsl_integrate_simplesed_r_r_d,
                          PARAMS->r_min,
                          PARAMS->r_max,
                          0,
                          PARAMS->epsrel,
                          w4,
                          &result,
                          &error,
                          &n_eval);
    result = 2. * PARAMS->z * result;
    return result;
}


void initialize_integrators()
{
    n_eval=50;
    gsl_integrate_simplesed_z_phi_d = malloc(sizeof(gsl_function));
    gsl_integrate_simplesed_z_phi_d->function = &integrate_simplesed_z_phi_d;
    gsl_integrate_simplesed_z_phi_d->params = NULL;

    gsl_integrate_simplesed_z_r_d = malloc(sizeof(gsl_function));
    gsl_integrate_simplesed_z_r_d->function = &integrate_simplesed_z_r_d;
    gsl_integrate_simplesed_z_r_d->params = NULL;

    gsl_integrate_simplesed_r_phi_d = malloc(sizeof(gsl_function));
    gsl_integrate_simplesed_r_phi_d->function = &integrate_simplesed_r_phi_d;
    gsl_integrate_simplesed_r_phi_d->params = NULL;

    gsl_integrate_simplesed_r_r_d = malloc(sizeof(gsl_function));
    gsl_integrate_simplesed_r_r_d->function = &integrate_simplesed_r_r_d;
    gsl_integrate_simplesed_r_r_d->params = NULL;

    w = gsl_integration_cquad_workspace_alloc(n_eval);
    w2 = gsl_integration_cquad_workspace_alloc(n_eval);
    w3 = gsl_integration_cquad_workspace_alloc(n_eval);
    w4 = gsl_integration_cquad_workspace_alloc(n_eval);
}

int main(void)
{
    parameters params;
    params.r = 100.;
    params.z = 50.;
    params.R_g = 1e14;
    params.astar = 0.;
    params.isco = 6.;
    params.r_min = 6.;
    params.r_max = 1600.;
    params.epsabs = 0.;
    params.epsrel = 1e-4;
    initialize_integrators();
    printf("%e\n",integrate_simplesed_z(&params));
    printf("%e\n",integrate_simplesed_r(&params));

    return 0;
}





