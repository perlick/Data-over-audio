#ifndef DOA_FILTER
#define DOA_FILTER
#include <complex.h>
#include "macros.h"

struct filter {
    int num_taps;
    float beta;
    int Ts;
    float *taps;
};
typedef struct filter Filter;

Filter *RootRaisedCosineFilter(float beta, float T, float ts);

Filter *create_filter_rrc1(float symbol_len, float beta, int Ts);

Filter *create_filter_rrc(int num_taps, float beta, float Ts);

Filter *create_filter_rc1(float symbol_len, float beta, float Ts);

Filter *create_filter_rc(int num_taps, float beta, float Ts);

fcomplex *convolve_valid(fcomplex *h, int lenH, Filter *x, int* lenY);

fcomplex *convolve(fcomplex *h, int lenH, Filter *x, int* lenY);

void save_filter(Filter* filter, const char *pathname);

#endif
