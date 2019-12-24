#include <math.h>
#include "disk.h"
double nt_rel_factors(double r, double astar, double isco)
{
    double yms, y1, y2, y3, y, C, B, A, factor;
    yms = sqrt(isco);
    y1 = 2 * cos((acos(astar) - M_PI) / 3);
    y2 = 2 * cos((acos(astar) + M_PI) / 3);
    y3 = -2 * cos(acos(astar) / 3);
    y = sqrt(r);
    C = 1 - 3 / r + 2 * astar / pow(r,1.5);
    B = 3 * pow(y1 - astar,2.) * log((y - y1) / (yms - y1)) / (y * y1 * (y1 - y2) * (y1 - y3));
    B += 3 * pow(y2 - astar,2.) * log((y - y2) / (yms - y2)) / (y * y2 * (y2 - y1) * (y2 - y3));
    B += 3 * pow(y3 - astar,2.) * log((y - y3) / (yms - y3)) / (y * y3 * (y3 - y1) * (y3 - y2));
    A = 1 - yms / y - 3 * astar * log(y / yms) / (2 * y);
    factor = (A - B) / C;
    return factor;
}
