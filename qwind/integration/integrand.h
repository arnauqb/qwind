#include <stdio.h>
typedef struct parameters{
    double r;
    double z;
    double r_d;
    double R_g;
    double astar;
    double isco;
    double r_min;
    double r_max;
    double epsabs;
    double epsrel;
} parameters;

double integrate_simplesed_z_phi_d(double phi_d, void *params);
double integrate_simplesed_z_r_d(double r_d, void *params);
double integrate_simplesed_z(parameters*);
double integrate_simplesed_r_phi_d(double phi_d, void *params);
double integrate_simplesed_r_r_d(double r_d , void *params);
double integrate_simplesed_r(parameters*);
void initialize_integrators(void);
