#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int get_arg(double x, double *arr, size_t n, double l, double r)
    /*Assumes the array is sorted.*/
{
    if (x < arr[0])
        return 0;
    if (x > arr[n-1])
        return n-1;

    if (r >= l) { 
        int mid = l + (r - l) / 2; 
        
        // If the element is present at the middle 
        // itself 
        if (arr[mid] == x) 
            return mid; 
        
        // If element is smaller than mid, then 
        // it can only be present in left subarray 
        if (arr[mid] > x) 
            return get_arg(x, arr, n, l, mid - 1); 
        
        // Else the element can only be present 
        // in right subarray 
        return get_arg(x, arr, n, mid + 1, r); 
    } 
    
    // We reach here when element is not 
    // present in array 
    return n-1; 
}
int sign(int x)
{
    if(x>0)
        return 1;
    else if(x<0)
        return -1;
    else
        return 0;
}
void drawline(int x1, int y1, int x2, int y2, int *results, int length)
{
    int x,y,dx,dy,swap,temp,s1,s2,p,i;

    x=x1;
    y=y1;
    dx=abs(x2-x1);
    dy=abs(y2-y1);
    s1=sign(x2-x1);
    s2=sign(y2-y1);
    swap=0;
    results[0] = x1;
    results[length] = y1;
    if(dy>dx)
    {
        temp=dx;
        dx=dy;
        dy=temp;
        swap=1;
    }
    p=2*dy-dx;
    for(i=0;i<dx;i++)
    {
        while(p>=0)
        {
            p=p-2*dx;
            if(swap)
                x+=s1;
            else
                y+=s2;
        }
        p=p+2*dy;
        if(swap)
            y+=s2;
        else
            x+=s1;
        results[i + 1] = x;
        results[length + i + 1] = y;
    }
    results[length - 1] = x2;
    results[2*length - 1] = y2;
}

double opacity_x_r(double xi)
{
    if (xi <= 1e5){
        return 100;
    }
    else{
        return 1;
    }
}

double tau_uv(double r, double z, int r_arg, int z_arg, double *density_grid, size_t n_z)
{
    int length, position; 
    length = fmax(r_arg, z_arg) + 1;
    int *results = malloc(2*length * sizeof(int));
    drawline(0, 0, r_arg, z_arg, results, length);
    double tau = 0;
    for (int i=0; i<length; i++)
    {
        position = results[i] * n_z + results[i + length];
        tau += density_grid[position];
    }
    tau = tau / length * sqrt(pow(r,2.) + pow(z,2.));
    free(results);
    return tau;
}

double tau_uv_disk_blob(double r_d, double phi_d, double r, double z, double *density_grid, double *grid_r_range, double *grid_z_range, size_t n_r, size_t n_z)
{
    double line_length;
    size_t r_arg, z_arg, r_d_arg, z_d_arg;
    double dr, dz;
    double tau = 0;
    int length, position; 
    line_length = sqrt(pow(r,2.) + pow(r_d,2.) + pow(z,2.) - 2 * r * r_d * cos(phi_d));
    r_arg = get_arg(r, grid_r_range, n_r, grid_r_range[0], grid_r_range[n_r-1]);
    z_arg = get_arg(z, grid_z_range, n_z, grid_z_range[0], grid_z_range[n_z-1]);
    r_d_arg = get_arg(r_d, grid_r_range, n_r, grid_r_range[0], grid_r_range[n_r-1]);
    z_d_arg = get_arg(0., grid_z_range, n_z, grid_z_range[0], grid_z_range[n_z-1]);
    dr = abs(r_arg - r_d_arg);
    dz = abs(z_arg - z_d_arg);
    length = fmax(dr, dz) + 1;
    int *results = malloc(2*length * sizeof(int));
    drawline(r_d_arg, z_d_arg, r_arg, z_arg, results, length);
    for (int i=0; i<length; i++)
    {
        position = results[i] * n_z + results[i + length];
        //printf("%d \n", position);
        tau += density_grid[position];
    }
    tau = tau / length * line_length;
    free(results);
    return tau;
}
void update_tau_x_grid(double *density_grid, double *ionization_grid, double *tau_x_grid, double *grid_r_range, double *grid_z_range, size_t n_r, size_t n_z)
{
    //int dr, dz;
    int length, position;
    double den, xi, r, z, tau;
    for (int i=0; i<n_r; i++){
        r = grid_r_range[i];
        for (int j=0; j<n_z; j++){
            length = fmax(i,j) + 1;
            int *results = malloc(2*length * sizeof(int));
            drawline(0, 0, i, j, results, length);
            tau = 0.;
            //printf("line\n");
            for (int k=0; k < length; k++){
                //printf("r_arg: %d \t z_arg %d \n", results[0][k], results[1][k]);
                //position = results[0][k] * n_z + results[1][k];
                position = results[k] * n_z + results[k + length];
                den = density_grid[position];
                xi = ionization_grid[position];
                tau += den * opacity_x_r(xi);
            }
            z = grid_z_range[j];
            tau = tau / length * sqrt(pow(r,2.) + pow(z,2.));
            tau_x_grid[i * n_z + j] = tau;
            free(results);
        }
    }
}


//int main()
//{
//    int x0, y0, x1, y1;
//    x0 = 0;
//    y0 = 0;
//    x1 = 2;
//    y1 = 10;
//
//    int length = 10 + 1;
//    int *results = malloc(2 * length * sizeof(int));
//    drawline(x0, y0, x1, y1, results, length);
//    for (int i =0; i < length-1; i++)
//    {
//        //printf("%d \n",results[i]);
//        printf("%d \t %d\n", results[i], results[i + length]);
//    }

    //size_t n_r, n_z;
    //n_r = length;
    //n_z = length+1;
    //int position;
    //double *grid_r_range = malloc(n_r * sizeof(double));
    //double *grid_z_range = malloc(n_z  * sizeof(double));
    //for (int i=0; i < n_r; i++)
    //{
    //    grid_r_range[i] = 0 + 100*i;
    //    grid_z_range[i] = 0 + 100*i;
    //}
    //double *density_grid = malloc((n_r * n_z) * sizeof(double));
    //double *ion_grid= malloc((n_r*n_z) * sizeof(double));
    //double *tau_x_grid = malloc((n_z*n_r) * sizeof(double));
    //printf("initializing grids...\n");
    //for (int i=0; i<n_r; i++)
    //{
    //    for (int j=0; j<n_z; j++)
    //    {
    //        position = i + n_z + j;
    //        density_grid[position] = 2e8;
    //        ion_grid[position] = 1e10;
    //        tau_x_grid[position] = 0.;
    //    }
    //}
    //printf("Done\n");
    //update_tau_x_grid(density_grid, ion_grid, tau_x_grid, grid_r_range, grid_z_range, n_r, n_z);
    //printf("done updating grid\n");
    //for (int i=0; i < length; i++){
    //    printf("\n");
    //    for (int j=0; j< (length + 1); j++){
    //        printf("%e\t",tau_x_grid[i*n_z + j]);

    //    }
    //}
//}
