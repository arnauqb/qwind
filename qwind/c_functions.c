#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>

double integrate_r_kernel_r_d( double , double * );
double integrate_r_kernel_phi_d( double , double *);
double integrate_z_kernel_r_d( double , double * );
double integrate_z_kernel_phi_d( double , double *);
void integrate( double , double, double, double *);
double distance_gas_disc(double , double , double , double );

void integrate( double r, double z, double tau_uv, double *result)
{
    double abs_uv = exp( - tau_uv ); 
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double result_r, error_r, result_z, error_z;
    double expected = -4.0;
    double parameters[] = {r,z};

    // radial integration 
    gsl_function F_r;
    F_r.function = &integrate_r_kernel_r_d;
    F_r.params = &parameters;
    gsl_integration_qags(&F_r, 6., 1400., 0., 1e-7,1000, w, &result_r, &error_r);
    //gsl_integration_cquad(&F, 6., 1400., 0., 1e-7, w, &result, &error, 1000);

    // z integration 
    gsl_function F_z;
    F_z.function = &integrate_z_kernel_r_d;
    F_z.params = &parameters;
    gsl_integration_qags(&F_z, 6., 1400., 0., 1e-7,1000, w, &result_z, &error_z);
    gsl_integration_workspace_free (w);

    result[0] = result_r * abs_uv;
    result[1] = result_z * abs_uv;
    return 0;
}

double integrate_r_kernel_r_d( double r_d, double  *params )
{
    
    double r = params[0];
    double z = params[1];

    double ff0 = (1. - sqrt( 6. / r_d )) / pow(r_d,3.);
    double darea = 2 * r_d;

    double parameters[] = {r_d, r, z};
    gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (1000);
    double result, error;
    gsl_function F2;
    F2.function = &integrate_r_kernel_phi_d;
    F2.params = &parameters;

    gsl_integration_qags(&F2, 0, 3.14159, 0, 1e-7, 1000, w2, &result, &error);
    //gsl_integration_cquad(&F2, 0, 3.14159, 0, 1e-7, w2, &result, &error, 1000);
    gsl_integration_workspace_free (w2);
    return result * ff0 * darea;
}

double integrate_r_kernel_phi_d( double phi_d, double *params)
{

    double r_d = params[0];
    double r = params[1];
    double z = params[2];
    double delta = distance_gas_disc(r_d, phi_d, r, z);
    double proj = z * ( r - r_d * cos(phi_d));
    double ff = proj / pow(delta, 4.);
    return ff;
}

double integrate_z_kernel_r_d( double r_d, double * params)
{
    double r = params[0];
    double z = params[1];

    double ff0 = (1. - sqrt( 6. / r_d )) / pow(r_d, 3.);
    double darea = 2 * r_d;

    double parameters[] = {r_d, r, z};
    gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (1000);
    double result, error;
    gsl_function F2;
    F2.function = &integrate_z_kernel_phi_d;
    F2.params = &parameters;

    gsl_integration_qags(&F2, 0, 3.14159, 0, 1e-7, 1000, w2, &result, &error);
    //gsl_integration_cquad(&F2, 0, 3.14159, 0, 1e-7, w2, &result, &error, 1000);
    gsl_integration_workspace_free (w2);
    return result * ff0 * darea;
}

double integrate_z_kernel_phi_d( double phi_d, double * params)
{
    double r_d = params[0];
    double r = params[1];
    double z = params[2];
    //printf("%f\t%f\t%f\n", r_d, r,z);
    double delta = distance_gas_disc(r_d, phi_d, r, z);
    double ff = pow(z,2.) / pow(delta,4.);
    //printf("%e\n",ff);
    return ff;
}
double distance_gas_disc(double r_d, double phi_d, double r, double z)
{
    double dist = pow(r,2.) + pow(r_d,2.) + pow(z,2.) - 2 * r * r_d * cos(phi_d);
    return dist;
}

int main()
{
    printf("test\n");
    double result[2];
    integrate(100., 200., 0, &result);
    printf("Ir : %e\nIz: %e \n", result[0], result[1]);
    return 0;
}