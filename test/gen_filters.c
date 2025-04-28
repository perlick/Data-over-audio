#include "../src/filter.h"
#include "../src/macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(){
    Filter *filter;
    //filter = RootRaisedCosineFilter(0.0f, 0.5 / 100, 1.0/8000);
    //save_filter(filter, "0.0_filter.f32");
    //filter = RootRaisedCosineFilter(0.2f, 0.5 / 100, 1.0/8000);
    //save_filter(filter, "0.2_filter.f32");
    //filter = RootRaisedCosineFilter(0.4f, 0.5 / 100, 1.0/8000);
    //save_filter(filter, "0.4_filter.f32");
    //filter = RootRaisedCosineFilter(0.6f, 0.5 / 100, 1.0/8000);
    //save_filter(filter, "0.6_filter.f32");
    //filter = RootRaisedCosineFilter(0.8f, 0.5 / 100, 1.0/8000);
    //save_filter(filter, "0.8_filter.f32");
    //filter = RootRaisedCosineFilter(1.0f, 0.5 / 100, 1.0/8000);
    //save_filter(filter, "1.0_filter.f32");

    //filter = create_filter_rrc1(80, 0.0f, 12);
    //save_filter(filter, "0.0_orig_filter.f32");
    //filter = create_filter_rrc1(80, 0.2f, 12);
    //save_filter(filter, "0.2_orig_filter.f32");
    //filter = create_filter_rrc1(80, 0.4f, 12);
    //save_filter(filter, "0.4_orig_filter.f32");
    //filter = create_filter_rrc1(80, 0.6f, 12);
    //save_filter(filter, "0.6_orig_filter.f32");
    //filter = create_filter_rrc1(80, 0.8f, 12);
    //save_filter(filter, "0.8_orig_filter.f32");
    filter = create_filter_rrc1(80, 0.35f, 12);
    save_filter(filter, "1.0_orig_filter.f32");

    int h_len = 800;
    fcomplex *h = calloc(h_len, sizeof(fcomplex));
    memset(h, 0, h_len*sizeof(fcomplex));
    h[0]   = (fcomplex) -1.0 + 0.0I;
    h[80]  = (fcomplex)  1.0 + 0.0I;
    h[160] = (fcomplex) -1.0 + 0.0I;
    h[240] = (fcomplex)  1.0 + 0.0I;
    h[320] = (fcomplex) -1.0 + 0.0I;
    h[400] = (fcomplex)  1.0 + 0.0I;
    h[480] = (fcomplex)  1.0 + 0.0I;
    h[560] = (fcomplex) -1.0 + 0.0I;
    h[640] = (fcomplex) -1.0 + 0.0I;
    h[720] = (fcomplex) -1.0 + 0.0I;
    FILE *file = fopen("test_samples.fc32", "w");
    fwrite(h, sizeof(fcomplex), h_len, file);
    fclose(file);

    // (h*f)*f
    fcomplex *s;
    int lenY;
    printf("filter taps: %d\n", filter->num_taps);
    s = convolve(h, h_len, filter, &lenY);

    file = fopen("test_convolution.fc32", "w");
    fwrite(s, sizeof(fcomplex), lenY, file);
    fclose(file);

    fcomplex *s2;
    int lenZ;
    s2 = convolve(s, lenY, filter, &lenZ);
    file = fopen("test_double_convolution.fc32", "w");
    fwrite(s2, sizeof(fcomplex), lenZ, file);
    fclose(file);

    // h*(f*f)
    fcomplex *f_p = calloc(filter->num_taps, sizeof(fcomplex));
    for (int i =0;i<filter->num_taps;i++)
        f_p[i] = filter->taps[i] + 0.0I;
    fcomplex *s3;
    int lenA;
    s3 = convolve(f_p, filter->num_taps, filter, &lenA);
    Filter *filter_2;
    filter_2 = create_filter_rrc(lenA, 0.35f, 80);
    for (int i =0;i<lenA;i++)
        filter_2->taps[i] = creal(s3[i]);
    file = fopen("test_assoc_1_convolution.f32", "w");
    fwrite(filter_2->taps, sizeof(float), filter_2->num_taps, file);
    fclose(file);

    fcomplex *s4;
    int lenB;
    s4 = convolve(h, h_len, filter_2, &lenB);
    file = fopen("test_assoc_2_convolution.fc32", "w");
    fwrite(s4, sizeof(fcomplex), lenB, file);
    fclose(file);
    free(h);
    free(s3);
    free(s2);
    free(s4);

}
