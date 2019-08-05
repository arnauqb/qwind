#include <stdio.h>
#include <math.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

double interpolate_point_grid_1d(double x, double *grid_values, size_t nx, double *x_range, double fill_value)
{
    /* Interpolates grid on a point, used to compute optical depths.*/
    if( x < x_range[0] || x > x_range[nx-1])
    {
        return fill_value;
    }
    const gsl_interp_type *T = gsl_interp_linear;

    gsl_spline *spline = gsl_spline_alloc(T, nx);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    size_t i;
    
    /* initialize interpolation */
    gsl_spline_init(spline, x_range, grid_values, nx);
    
    double result = gsl_spline_eval(spline, x, xacc);
  
    gsl_spline_free(spline);
    gsl_interp_accel_free(xacc);
    return result;
}
