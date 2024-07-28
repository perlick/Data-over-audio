#include "filter.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

Filter *create_filter_rrc1(float symbol_len, float beta, float Ts){
    int num_taps = (int) (symbol_len * Ts); 
    if (num_taps%2==1)
        num_taps+=1;
    return create_filter_rrc(num_taps, beta, Ts);
}

Filter *create_filter_rrc(int num_taps, float beta, float Ts){
    Filter *filt = malloc(sizeof(Filter));
    filt->num_taps = num_taps;
    filt->beta = beta;
    filt->Ts=Ts;
    filt->taps = calloc(num_taps, sizeof(float));

    float t;
    float tap, x;
    for (int i=0;i<num_taps;i++){
        t = i - (num_taps-1)/2;
        if (t==0){
            tap = 1;
        }else if(t==abs(Ts/(4*beta))){
            tap = beta/(Ts*sqrt(2)) * ((1+(2/M_PI))*sin(M_PI/(4*beta)) + (1+(2/M_PI))*cos(M_PI/(4*beta)));
        } else {
            tap = (1/Ts) * (sin(M_PI*(t/Ts)*(1-beta)) + 4*beta*(t/Ts)*cos(M_PI*(t/Ts)*(1+beta))) / (M_PI*(t/Ts)*(1-(4*beta*(t/Ts))*(4*beta*(t/Ts))));
        }
        filt->taps[i] = tap;
    }
    FILE *filter_cap = fopen("filter.f32", "w");
    fwrite(filt->taps, sizeof(float), num_taps, filter_cap);
    fclose(filter_cap);
    return filt;
}

Filter *create_filter_rc1(float symbol_len, float beta, float Ts){
    int num_taps = (int) (symbol_len * Ts); 
    if (num_taps%2==1)
        num_taps+=1;
    return create_filter_rc(num_taps, beta, Ts);
}

Filter *create_filter_rc(int num_taps, float beta, float Ts){
    Filter *filt = malloc(sizeof(Filter));
    filt->num_taps = num_taps;
    filt->beta = beta;
    filt->Ts=Ts;
    filt->taps = calloc(num_taps, sizeof(float));

    float t;
    float tap, x;
    for (int i=0;i<num_taps;i++){
        t = i - (num_taps-1)/2;
        x = t/Ts;
        if (x !=0 ){
            tap = sin(M_PI*x)/(M_PI*x) * cos(M_PI*beta*x) / (1 - (2*beta*t/Ts)*(2*beta*t/Ts));
        }else{
            tap = 1;
        }
        filt->taps[i] = tap;
    }
    //FILE *filter_cap = fopen("filter.f32", "w");
    //fwrite(filt->taps, sizeof(float), num_taps, filter_cap);
    //fclose(filter_cap);
    return filt;
}

float complex *convolve(float complex *h, int lenH, Filter *filter, int* lenY){
    int lenX = filter->num_taps;
    int nconv = lenH+lenX-1;
    *lenY = nconv;
    int i,j,h_start,x_start,x_end;

    float complex *y = calloc(nconv, sizeof(float complex));

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

