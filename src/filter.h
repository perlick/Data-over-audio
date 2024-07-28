#ifndef DOA_FILTER
#define DOA_FILTER
#include <complex.h>

struct filter {
    int num_taps;
    float beta;
    int Ts;
    float *taps; 
};
typedef struct filter Filter;

Filter *create_filter_rrc1(float symbol_len, float beta, float Ts);

Filter *create_filter_rrc(int num_taps, float beta, float Ts);

Filter *create_filter_rc1(float symbol_len, float beta, float Ts);

Filter *create_filter_rc(int num_taps, float beta, float Ts);

float complex *convolve(float complex *h, int lenH, Filter *x, int* lenY);

#endif
