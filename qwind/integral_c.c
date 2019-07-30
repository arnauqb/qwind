#include <math.h>
#include <stdio.h>

double delta_gas_disk(double r_d, double phi_d, double r, double z )
{
    double delta = sqrt( pow(r,2.) + pow(z,2.) + pow(r_d,2.) - 2 * r * r_d * cos(phi_d));
    return delta;
}

double non_adaptive_integral_r( double r, double z, double r_min, double r_max, int N_PHI, int N_R)
{
    double r_d;
    double r_d_log_0;
    double r_d_log_1;
    double delta_r;
    double phi_d;
    double aux;
    double delta_log_r = (log10(r_max) - log10(r_min)) / N_R;
    double delta_phi = M_PI / N_PHI;

    double integral = 0.;
    double integral_phi = 0;
    for (int i = 0; i < N_R; i++)
    {
        r_d_log_0 = log10(r_min) + delta_log_r * i;
        r_d_log_1 = log10(r_min) + delta_log_r * (i+1);
        r_d = pow(10., r_d_log_0); 
        delta_r = pow(10., r_d_log_1) - r_d;
        integral_phi = 0;
        for (int j = 0; j < N_PHI; j++)
        {
            phi_d = j * delta_phi; 
            aux = (r - r_d * cos(phi_d)) / pow(delta_gas_disk(r_d, phi_d, r, z), 4.);
            integral_phi += aux * delta_phi;
        }
        aux = (1. - sqrt(6. / r_d )) / pow(r_d, 2.);
        integral += aux * delta_r * integral_phi;
        //printf("%e \t %e \t %e\n", r, r_d, z);
        //printf("integral_phi: %e \n",integral * 2. * pow(z,2.));
    }
    integral = 2. * z * integral;
    return integral;
}

double non_adaptive_integral_z( double r, double z, double r_min, double r_max, int N_PHI, int N_R)
{
    double r_d;
    double r_d_log_0;
    double r_d_log_1;
    double delta_r;
    double phi_d;
    double aux;
    double delta_log_r = (log10(r_max) - log10(r_min)) / N_R;
    double delta_phi = M_PI / N_PHI;

    double integral = 0.;
    double integral_phi = 0;
    for (int i = 0; i < N_R; i++)
    {
        r_d_log_0 = log10(r_min) + delta_log_r * i;
        r_d_log_1 = log10(r_min) + delta_log_r * (i+1);
        r_d = pow(10., r_d_log_0); 
        delta_r = pow(10., r_d_log_1) - r_d;
        integral_phi = 0;
        for (int j = 0; j < N_PHI; j++)
        {
            phi_d = j * delta_phi; 
            aux = 1. / pow(delta_gas_disk(r_d, phi_d, r, z), 4.);
            integral_phi += aux * delta_phi;
        }
        aux = (1. - sqrt(6. / r_d )) / pow(r_d, 2.);
        integral += aux * delta_r * integral_phi;
        //printf("%e \t %e \t %e\n", r, r_d, z);
        //printf("integral_phi: %e \n",integral * 2. * pow(z,2.));
    }
    integral = 2. * pow(z,2.) * integral;
    return integral;
}

double adaptive_integral_r

int main()
{
    double a = non_adaptive_integral_r(236.842, 1.000, 6., 1400., 100, 250);
    double b = non_adaptive_integral_z(236.842, 1.000, 6., 1400., 100, 250);
    printf("Ir: %e\n",a);
    printf("Iz: %e\n",b);
    return 0;
}