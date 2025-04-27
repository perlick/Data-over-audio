#include "filter.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include <string.h>

double sinc(float x) {
  if (x == 0.0) {
    return 1.0;
  } else {
    return sin(x) / x;
  }
}


/* Create root raised cosine filter

beta:  damping factor
T:     Half of the symbol duration in seconds
ts:    time per sample in seconds (reciprocal or sample rate)
*/
Filter *RootRaisedCosineFilter(float beta, float T, float ts)
{
    Filter *filt = malloc(sizeof(Filter));
    filt->beta = beta;

    float t;
    const int Nsymb = 12;
    const float samp_per_symb = T/ts;
    filt->Ts=samp_per_symb;
    unsigned long N = (unsigned long) Nsymb*samp_per_symb+1;
    filt->num_taps = N;
    filt->taps = calloc(N, sizeof(float));

    float max = 0.0;
    float shift = -Nsymb*T/2.0;
    for(int i=0;i<N;i++)
    {
        t = shift+ts*((float) i);

        if(fabs(t) == T/2.0/beta)
        {
            filt->taps[i] = M_PI*sinc(1.0/2.0/beta)/Nsymb/T/2.0;
            continue;
        }
        float tv = t/T;
        float tvb = beta*tv;
        float t1 = sinc(tv)/T;
        float t2 = cos(M_PI*tvb);
        float t3 = 1-pow(2.0*tvb,2);
        filt->taps[i] = t1*t2/t3;
        if(filt->taps[i]>max)
        {
            max = filt->taps[i];
        }
    }
    //for(int i=0;i<N;i++) filt->taps[i] /= max;

    FILE *filter_cap = fopen("filter.f32", "w");
    fwrite(filt->taps, sizeof(float), N, filter_cap);
    fclose(filter_cap);
    return filt;
}

/* Create a root-raised cosine filter.

symbol_len: number of samples per symbol
beta: lowering beta will lower bandwidth usage and increase filter tails.
Ts: number of symbols over which filter should apply
*/
Filter *create_filter_rrc1(float symbol_len, float beta, int Ts){
    int num_taps = (int) (symbol_len * Ts);
    if (num_taps%2==1)
        num_taps+=1;
    return create_filter_rrc(num_taps, beta, symbol_len);
}

/* Create a root-raised cosine filter.

num_taps: number of taps in filter
beta: lowering beta will lower bandwidth usage and increase filter tails.
Ts: number of samples per symbol
*/
Filter *create_filter_rrc(int num_taps, float beta, float Ts){
    Filter *filt = malloc(sizeof(Filter));
    filt->num_taps = num_taps;
    filt->beta = beta;
    filt->Ts=Ts;
    filt->taps = calloc(num_taps, sizeof(float));

    float t;
    float tap, x, max, scale;
    max = 0;
    for (int i=0;i<num_taps;i++){
        t = i - (num_taps-1)/2;
        if (t==0){
            tap = (1/Ts) * (1+beta*((4/M_PI)-1));
        }else if(abs(t)==abs(Ts/(4*beta))){
            tap = beta/(Ts*sqrt(2)) * ((1+(2/M_PI))*sin(M_PI/(4*beta)) + (1-(2/M_PI))*cos(M_PI/(4*beta)));
        } else {
            tap = (1/Ts) * (sin(M_PI*(t/Ts)*(1-beta)) + 4*beta*(t/Ts)*cos(M_PI*(t/Ts)*(1+beta))) / (M_PI*(t/Ts)*(1-(4*beta*(t/Ts))*(4*beta*(t/Ts))));
        }
        filt->taps[i] = tap;
        if(fabs(tap) > max)
            max = fabs(tap);
    }
    for (int i=0;i<num_taps;i++)
        filt->taps[i] = filt->taps[i] * 0.6 / max ;

    FILE *filter_cap = fopen("filter.f32", "w");
    fwrite(filt->taps, sizeof(float), num_taps, filter_cap);
    fclose(filter_cap);
    return filt;
}

Filter *create_filter_rc1(float symbol_len, float beta, float Ts){
    int num_taps = (int) (symbol_len * Ts);
    if (num_taps%2==1)
        num_taps+=1;
    return create_filter_rc(num_taps, beta, symbol_len);
}

Filter *create_filter_rc(int num_taps, float beta, float Ts){
    Filter *filt = malloc(sizeof(Filter));
    filt->num_taps = num_taps;
    filt->beta = beta;
    filt->Ts=Ts;
    filt->taps = calloc(num_taps, sizeof(float));

    float t;
    float tap, x, max, scale;
    max = 0;
    for (int i=0;i<num_taps;i++){
        t = i - (num_taps-1)/2;
        x = t/Ts;
        if (x !=0 ){
            tap = sinc(x) * cos(M_PI*beta*x) / (1 - (2*beta*t/Ts)*(2*beta*t/Ts));
        }else{
            tap = 1;
        }
        filt->taps[i] = tap;
        if(fabs(tap) > max)
            max = fabs(tap);
    }
    //for (int i=0;i<num_taps;i++)
    //    filt->taps[i] = filt->taps[i] * 0.6 / max ;

    FILE *filter_cap = fopen("filter.f32", "w");
    fwrite(filt->taps, sizeof(float), num_taps, filter_cap);
    fclose(filter_cap);
    return filt;
}

fcomplex *convolve_valid(fcomplex *h, int lenH, Filter *filter, int* lenY){
    int lenX = filter->num_taps;
    int nconv = fmax(lenH,lenX) - fmin(lenH,lenX) + 1;
    *lenY = nconv;
    int i,j,h_start,x_start,x_end;

    fcomplex *y = calloc(nconv, sizeof(fcomplex));
    memset(y, 0, nconv*sizeof(fcomplex));

    for (i=0; i<nconv; i++){
        x_start = 0;
        x_end = lenX;
        h_start = i;
        for(j=x_start; j<x_end; j++){
              y[i] += h[h_start++]*filter->taps[j];
        }
    }
    return y;
}

/* Convolve signal and filter.

h: signal
lenH: length of H in elements
filter: filter to be used
lenY: length of returned signal
*/
fcomplex *convolve(fcomplex *h, int lenH, Filter *filter, int* lenY){
    int lenX = filter->num_taps;
    int nconv = lenH+lenX-1;
    *lenY = nconv;
    int i,j,h_start,x_start,x_end;

    fcomplex *y = calloc(nconv, sizeof(fcomplex));
    memset(y, 0, nconv*sizeof(fcomplex));

    for (i=0; i<nconv; i++){
        x_start = fmax(0,i-lenH+1);
        x_end   = fmin(i+1,lenX);
        h_start = fmin(i,lenH-1);
        for(j=x_start; j<x_end; j++){
              y[i] += h[h_start--]*filter->taps[j];
        }
    }
    return y;
}

/* Save a Filter to file

filter: filter to be saved
pathname: path to save to
*/

void save_filter(Filter* filter, const char *pathname){
    FILE *filter_cap = fopen(pathname, "w");
    fwrite(filter->taps, sizeof(float), filter->num_taps, filter_cap);
    fclose(filter_cap);
}

