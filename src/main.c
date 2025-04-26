#include <complex.h>
#include <errno.h>
#include <fftw3.h>
#include <math.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef _WIN32
#include <windows.h>
#include <synchapi.h>
#elif __linux
#include "linux_asoundlib.h"
#endif
#include "macros.h"
#include "circ_buf.h"
#include "filter.h"
#include "shared_mem.h"
#include "proc.h"
#include "mcs.h"


#define noop

static int max_L2_packet_size_bytes = 1500;


int intPow(int x,int n)
{
    int i; /* Variable used in loop counter */
    int number = 1;

    for (i = 0; i < n; ++i)
        number *= x;

    return(number);
}

void tx_encode_packet(CircBuf *buf, MCS *mcs, CircBuf *out_buf, FILE *plbk_sym){
    // buffer for raw data
    char* data_buf = malloc(max_L2_packet_size_bytes);
    // buffer for symbols
    int symbol_buf_len = ((max_L2_packet_size_bytes*8) / mcs->bits_per_symbol) + 1;
    fcomplex *symbol_buf = malloc(symbol_buf_len);
    // buffer for upscaled symbols
    if (mcs->output_sample_rate_hz % mcs->symbol_rate_hz != 0){
    	printf("output_sample_rate_hz must be divisible by symbol_rate_hz!");
        exit(1);
    }
    int samples_per_symbol = mcs->input_sample_rate_hz / mcs->symbol_rate_hz;
    int sample_buf_len = symbol_buf_len * samples_per_symbol;
    fcomplex *sample_buf = calloc(sample_buf_len, sizeof(fcomplex));
    for (int i = 0; i < sample_buf_len; i++)
        sample_buf[i] = 0;

    // read data from circular buffer
    int num_read = read_buf(buf, max_L2_packet_size_bytes, data_buf, 0);

    // channel coding
    if (mcs->channel_coding==0){
        noop;
    }

    // convert to symbols
    int num_symbols = (((num_read*8) + mcs->bits_per_symbol - 1) / mcs->bits_per_symbol);
    // printf("num symbols: %d\n", num_symbols);
    int mask = intPow(2,mcs->bits_per_symbol) - 1;
    for(int i=0;i<num_symbols;i=i+1){
        int first_bit_index = i * mcs->bits_per_symbol;
        int first_byte_index = first_bit_index / 8;
        int last_bit_index = first_bit_index + mcs->bits_per_symbol;
        int last_byte_index = last_bit_index / 8;
        int sym_int;
        if(first_byte_index == last_byte_index){
            sym_int = data_buf[first_byte_index];
        }else if(first_byte_index == last_byte_index+1){
            sym_int = data_buf[first_byte_index] + (data_buf[last_byte_index]<<8);
        }else if(first_byte_index == last_byte_index+2){
            sym_int = data_buf[first_byte_index] + (data_buf[first_byte_index+1]<<8) + (data_buf[last_byte_index]<<16);
        }else if(first_byte_index == last_byte_index+3){
            sym_int = data_buf[first_byte_index] + (data_buf[first_byte_index+1]<<8) + (data_buf[first_byte_index+2]<<16) + (data_buf[last_byte_index]<<24);
        }
        int shift = first_bit_index % 8;
        sym_int = sym_int >> shift & mask;
        // TODO convert this to a dict lookup
        for(int j=0;j<mcs->num_symbols;j=j+1){
            if(sym_int == mcs->symbol_list_int[j]){
                symbol_buf[i] = mcs->symbol_list_complex[j];
                break;
            }
        }
        //printf("symbol %d: %f + i%f\n", i, creal(symbol_buf[i]), cimag(symbol_buf[i]));
    }

    // do pulse shaping with matched filter. upscaling by samples per symbol
    int num_samples;
    num_samples = num_symbols * samples_per_symbol;
    for(int i=0;i<num_symbols;i++){
        sample_buf[i*samples_per_symbol] = symbol_buf[i];
    }
    fwrite(sample_buf, sizeof(fcomplex), num_samples, plbk_sym);

    int len_filt_buf;
    fcomplex *filt_buf = convolve_valid(sample_buf, num_samples, mcs->tx_filter, &len_filt_buf);

    // write full packet to front end buffer in one shot.
    int count;
    printf("len_sample_buf: %d\n", num_samples);
    fflush(stdout);
    while ((count = write_buf(filt_buf, len_filt_buf, out_buf, 1)) == 0)
        sleep(0.01);

    free(sample_buf);
    free(symbol_buf);
    free(data_buf);
    printf("finished encoding samples: %d\n", len_filt_buf);
    fflush(stdout);
    return;
}

int main(){
    /* setup bsaic handlers/ */
    // struct sigaction sa;
    // sa.sa_handler = handler;
    // sigemptyset(&sa.sa_mask);
    // sa.sa_flags = 0;

    // if (sigaction(SIGINT, &sa, NULL) == -1){
    //     printf("failed to register signal handler SIGINT: %s\n", strerror(errno));
    //     exit(1);
    // }

    char myArray[] = { 0xff, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12,0x00, 0x11, 0x22, 0xff, 0xec, 0x12};

    char* tx_input_buffer = malloc(max_L2_packet_size_bytes);

    struct circBuf in_buf;
    in_buf.element_size = sizeof(char);
    in_buf.start = tx_input_buffer;
    in_buf.len = max_L2_packet_size_bytes;
    in_buf.read_idx = 0;
    in_buf.write_idx = 0;
    in_buf.count = 0;
    in_buf.stream = NULL;

    int x = write_buf(myArray, 24, &in_buf, 1);

    /* bpsk */
    struct mcs mcs0;
    mcs0.channel_coding = 0;
    mcs0.bits_per_symbol = 1;
    mcs0.num_symbols = 2;
    mcs0.symbol_list_int = malloc(mcs0.num_symbols);
    mcs0.symbol_list_complex = malloc(mcs0.num_symbols * sizeof(fcomplex));
    mcs0.symbol_list_int[0] = 0;
    mcs0.symbol_list_complex[0] = (fcomplex) 1.0 + 0.0I;
    mcs0.symbol_list_int[1] = 1;
    mcs0.symbol_list_complex[1] = (fcomplex) -1.0 + 0.0I;
    mcs0.output_sample_rate_hz = 8000;
    mcs0.symbol_rate_hz = 100;
    mcs0.carrier_freq_hz = 440;
    mcs0.input_sample_rate_hz = 8000;
    mcs0.order = 2;
    mcs0.mnm_aggression = 0.35f;
    mcs0.tx_filter = create_filter_rrc1((float) mcs0.output_sample_rate_hz / mcs0.symbol_rate_hz, 0.35f, 12.625f);
    mcs0.rx_filter = create_filter_rrc1((float) mcs0.input_sample_rate_hz / mcs0.symbol_rate_hz, 0.35f, 12.625f);

    struct mcs mcs1;
    mcs1.channel_coding = 0;
    mcs1.bits_per_symbol = 2;
    mcs1.num_symbols = 4;
    mcs1.symbol_list_int = malloc(mcs1.num_symbols);
    mcs1.symbol_list_complex = malloc(mcs1.num_symbols * sizeof(fcomplex));
    mcs1.symbol_list_int[0] = 0;
    mcs1.symbol_list_complex[0] = (fcomplex) 1.0 + 0.0I;
    mcs1.symbol_list_int[1] = 1;
    mcs1.symbol_list_complex[1] = (fcomplex) -1.0 + 0.0I;
    mcs1.symbol_list_int[2] = 2;
    mcs1.symbol_list_complex[2] = (fcomplex) 0.0 + 1.0I;
    mcs1.symbol_list_int[3] = 3;
    mcs1.symbol_list_complex[3] = (fcomplex) 0.0 + -1.0I;
    mcs1.output_sample_rate_hz = 8000;
    mcs1.symbol_rate_hz = 100;
    mcs1.carrier_freq_hz = 440;
    mcs1.input_sample_rate_hz = 8000;
    mcs1.order = 4;
    mcs1.mnm_aggression = 0.3f;
    mcs1.tx_filter = create_filter_rrc1((float) mcs1.output_sample_rate_hz / mcs1.symbol_rate_hz, 0.35f, 12.625f);
    mcs1.rx_filter = create_filter_rrc1((float) mcs1.input_sample_rate_hz / mcs1.symbol_rate_hz, 0.35f, 12.625f);

    struct mcs *cur_mcs = &mcs0;

    // put the fe buffer in a shared memory space.
    int samples_per_symbol = cur_mcs->output_sample_rate_hz / cur_mcs->symbol_rate_hz;
    int max_L2_packet_size_samples = (max_L2_packet_size_bytes * 8 / cur_mcs->bits_per_symbol) * samples_per_symbol + cur_mcs->tx_filter->num_taps + 1;
    char* fe_input_buffer = create_shared_memory(max_L2_packet_size_samples * sizeof(fcomplex));
    struct circBuf *fe_buf = create_shared_memory(sizeof(struct circBuf));
    fe_buf->element_size = sizeof(fcomplex);
    fe_buf->start = fe_input_buffer;
    fe_buf->len = max_L2_packet_size_samples;
    fe_buf->read_idx = 0;
    fe_buf->write_idx = 0;
    fe_buf->count = 0;
    fe_buf->stream = fopen("plbk_2_iq.fc32", "w");
    //fe_buf->stream = NULL;
    char* sample_input_buffer = create_shared_memory(10000 * sizeof(short));
    struct circBuf *sample_buf = create_shared_memory(sizeof(struct circBuf));
    sample_buf->element_size = sizeof(short);
    sample_buf->start = sample_input_buffer;
    sample_buf->len = 10000;
    sample_buf->read_idx = 0;
    sample_buf->write_idx = 0;
    sample_buf->count = 0;
    sample_buf->stream = NULL; // set this in subproc

    FILE *plbk_sym = fopen("plbk_1_sym.fc32", "w");

    spawn_rx_chain(cur_mcs, sample_buf);

    // convert a packet of data to IQ symbols
    tx_encode_packet(&in_buf, cur_mcs, fe_buf, plbk_sym);

    x = write_buf(myArray, 24, &in_buf, 1);
    tx_encode_packet(&in_buf, cur_mcs, fe_buf, plbk_sym);
    x = write_buf(myArray, 24, &in_buf, 1);
    tx_encode_packet(&in_buf, cur_mcs, fe_buf, plbk_sym);
    x = write_buf(myArray, 24, &in_buf, 1);
    tx_encode_packet(&in_buf, cur_mcs, fe_buf, plbk_sym);

    spawn_tx_chain(cur_mcs, fe_buf, sample_buf);

    char *line = NULL;
    size_t size;
    size_t num;
    while (1){
        num = getline(&line, &size, stdin);
        if (num == -1) {
            printf("No line\n");
        } else {
            x = write_buf(line, num, &in_buf, 1);
            tx_encode_packet(&in_buf, &mcs1, fe_buf, plbk_sym);
        }
    }
    fclose(plbk_sym);
};

