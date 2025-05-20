#ifndef DOA_MCS
#define DOA_MCS
#include "filter.h"

struct mcs {
    int channel_coding; // 0 is none
    int bits_per_symbol; // Number of bits per symbol
    int num_symbols; // Number of symbols / len of lists
    char *symbol_list_int; //Array of integer symbols. data representation
    fcomplex *symbol_list_complex; // Parallel array of complex numbers. signal representation
    int output_sample_rate_hz; // rate of the samples being sent to hardware device
    int symbol_rate_hz; // Rate of symbols being encoded into signal
    int carrier_freq_hz; // Tune Lo to this freq
    int input_sample_rate_hz;
    int order; // Course frequency estimate multiplies by this number before taking FFT
    float mnm_aggression;
    Filter *tx_filter;
    Filter *rx_filter;
    fcomplex *frame_detect_signal;
    int len_frame_detect_signal;
    float frame_detect_level;
};
typedef struct mcs MCS;

#endif
