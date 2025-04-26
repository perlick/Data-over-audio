#include "../src/filter.h"
#include <stdlib.h>

int main(){
    Filter *filter;
    filter = RootRaisedCosineFilter(0.0f, 0.5 / 100, 1.0/8000);
    save_filter(filter, "0.0_filter.f32");
    filter = RootRaisedCosineFilter(0.2f, 0.5 / 100, 1.0/8000);
    save_filter(filter, "0.2_filter.f32");
    filter = RootRaisedCosineFilter(0.4f, 0.5 / 100, 1.0/8000);
    save_filter(filter, "0.4_filter.f32");
    filter = RootRaisedCosineFilter(0.6f, 0.5 / 100, 1.0/8000);
    save_filter(filter, "0.6_filter.f32");
    filter = RootRaisedCosineFilter(0.8f, 0.5 / 100, 1.0/8000);
    save_filter(filter, "0.8_filter.f32");
    filter = RootRaisedCosineFilter(1.0f, 0.5 / 100, 1.0/8000);
    save_filter(filter, "1.0_filter.f32");
    filter = create_filter_rrc1(80, 0.0f, 12);
    save_filter(filter, "0.0_orig_filter.f32");
    filter = create_filter_rrc1(80, 0.2f, 12);
    save_filter(filter, "0.2_orig_filter.f32");
    filter = create_filter_rrc1(80, 0.4f, 12);
    save_filter(filter, "0.4_orig_filter.f32");
    filter = create_filter_rrc1(80, 0.6f, 12);
    save_filter(filter, "0.6_orig_filter.f32");
    filter = create_filter_rrc1(80, 0.8f, 12);
    save_filter(filter, "0.8_orig_filter.f32");
    filter = create_filter_rrc1(80, 1.0f, 12);
    save_filter(filter, "1.0_orig_filter.f32");
}
