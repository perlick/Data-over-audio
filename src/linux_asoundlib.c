#include "linux_asoundlib.h"
#include <alsa/asoundlib.h>
#include <complex.h>
#include <errno.h>
#include <fftw3.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include "circ_buf.h"
#include "macros.h"
#include "mcs.h"
#include "filter.h"

static char *device = "default";         /* playback device */
static unsigned int channels = 1;           /* count of channels */
static unsigned int rate;           /* stream rate */
static unsigned int buffer_time = 2e3;       /* ring buffer length in us */
static unsigned int period_time = 1e3;       /* period time in us */
static int method = 1;
static double freq;               /* sinusoidal wave frequency in Hz */
static int verbose = 0;                 /* verbose flag */
static int resample = 0;                /* enable alsa-lib resampling */
static int period_event = 1;                /* produce poll event after each period */

static snd_pcm_format_t format = SND_PCM_FORMAT_S16;    /* sample format */
static snd_pcm_sframes_t buffer_size;
static snd_pcm_sframes_t period_size;

static snd_output_t *output = NULL;

int format_bits = 16;
unsigned int maxval = 32767; //(1 << (format_bits - 1)) - 1;
int bps = 2;  /* bytes per sample */
int phys_bps = 2;
int big_endian = 0;
int to_unsigned = 0;
int is_float = 0;

struct transfer_method {
    const char *name;
    snd_pcm_access_t access;
    int (*transfer_loop)(snd_pcm_t *handle,
                 signed short *samples,
                 snd_pcm_channel_area_t *areas,
                 CircBuf *iq_buf, int lo_freq);
};

float symbol_distance(fcomplex s1, fcomplex s2){
   return sqrt((creal(s2) - creal(s1))*(creal(s2) - creal(s1)) + (cimag(s2) - cimag(s1))*(cimag(s2) - cimag(s1)));
}

static void run_front_end_calculation(
              CircBuf *sample_buf, /*output sample buffer*/
              int count, /*number of iq samples to be converted form iq_buf*/
              double *_phase, /*lo phase at time of the first sample*/
              CircBuf *iq_buf, /*iq samples to be mixed*/
              int lo_freq, /*frequency of lo*/
              int sample_rate_hz, /*output sample rate in hz*/
              FILE *file /*output file*/
        ){
    //printf("Front end: lo_freq(%d), rate(%d)\n", lo_freq, rate);
    static double max_phase = 2.0 * M_PI;
    double phase = *_phase;
    double step = max_phase*lo_freq/(double)sample_rate_hz;
    fcomplex sample;
    int num_read;
    int num_written;

    while (count-- > 0) {
        // read a sample from the buffer
        num_read = read_buf(iq_buf, 1, &sample, 0);
        // if there's nothing to read, play nothing
        if (num_read == 0)
            sample = 0 + 0*I;

        short res, i;
        float inter;
        // Assumes amplitudes of I and Q do not exceed -1,1
        inter = creal(sample) * sin(phase) + cimag(sample) * cos(phase);
        inter = fmin(inter, 1);
        inter = fmax(inter, -1);
        inter += (rand() / (float) RAND_MAX) * 0.25;
        res = inter * maxval;
        while (write_buf(&res, 1, sample_buf, 1) == 0)
            sleep(0.01);
        //fwrite(&res, bps, 1, file);
        //printf("fe calc: I(%f) * sin(%f) + Q(%f) * cos(%f) * maxval(%d) = res(%d); \n", creal(sample), phase, cimag(sample), phase, maxval, res);
        phase += step;
        if (phase >= max_phase)
            phase -= max_phase;
    }
    *_phase = phase;
}

void start_tx_chain(
        MCS *mcs,
        CircBuf *iq_buf,
        CircBuf *sample_buf
        ){
    double phase = 0;
    int period_size = 128; /*not sure if this really matters for the loopback device*/
    int lo_freq = mcs->carrier_freq_hz + 1;
    //FILE *plbk_raw = fopen("plbk_3_raw.s16", "w");
    FILE *plbk_raw = NULL;
    sample_buf->stream = fopen("plbk_3_raw.s16", "w");

    for (int i=0;i<500;i++){
        run_front_end_calculation(sample_buf, period_size, &phase, iq_buf, lo_freq, mcs->output_sample_rate_hz, plbk_raw);
    }

    return;
}


void start_rx_chain(
        MCS *mcs,
        CircBuf *sample_c_buf
    ){
    // read data from input buffer
    int i;
    int err;
    int buf_size = 2048;
    int16_t buf[buf_size];
    int rate = mcs->input_sample_rate_hz;
    int lo_freq = mcs->carrier_freq_hz;


    FILE *file_raw = fopen("cap_1_raw.s16", "w");
    FILE *file_iq = fopen("cap_2_iq.fc32", "w");
    FILE *file_flt = fopen("cap_3_flt.fc32", "w");
    FILE *file_course_fft = fopen("cap_4_course.fft", "w");
    FILE *file_course = fopen("cap_4_course.fc32", "w");
    FILE *file_mnm = fopen("cap_5_mnm.fc32", "w");
    FILE *file_mnm_log = fopen("cap_5_mnm_log.f32", "w");
    FILE *file_ffs = fopen("cap_6_ffs.fc32", "w");
    FILE *file_ffs_ofst = fopen("cap_6_ffs_log.f3c32", "w");
    FILE *file_const = fopen("cap_7_const.const", "w");
    FILE *file_demod_sym = fopen("cap_8_demod_sym.fc32", "w");
    //printf("Front end: lo_freq(%d), rate(%d)\n", lo_freq, rate);
    static double max_phase = 2.0 * M_PI;
    double phase = 0;
    double step = max_phase*lo_freq/(double)rate;
    fcomplex sample_buf[buf_size];
    int filt_in_buf_size = buf_size + mcs->rx_filter->num_taps - 1;
    fcomplex filt_in_buf[filt_in_buf_size];
    memset(filt_in_buf, 0, sizeof(fcomplex)*filt_in_buf_size);
    fcomplex *filt_out_buf;
    double complex freq_est_in_buf[buf_size];
    float scaled;
    int order = mcs->order;
    fftw_complex fft_buf[buf_size];
    fftw_plan plan = fftw_plan_dft_1d(buf_size, freq_est_in_buf, fft_buf, FFTW_FORWARD, FFTW_MEASURE);
    int max;
    float freq_offset_est_hz;
    float mu = 0;
    fcomplex out[buf_size + 10];
    fcomplex out_rail[buf_size + 10];
    memset(out_rail, 0, sizeof(fcomplex)*(buf_size + 10));
    int i_in = 0;
    int i_out;
    float mm_val, real, imag;
    fcomplex x, y;
    float mnm_log[buf_size*2];
    int samples_per_symbol = mcs->input_sample_rate_hz / mcs->symbol_rate_hz;
    float costas_phase = 0;
    float freq = 0;
    float error;
    fcomplex costas_out[buf_size];
    float freq_log[buf_size*2];
    char bits[buf_size/(mcs->bits_per_symbol*8)];
    while (1) {
        /* Get Raw Samples */
        while (read_buf(sample_c_buf, buf_size, buf, 1) == 0)
            sleep(0.01);
        fwrite(buf, sizeof(int16_t), buf_size, file_raw);
        fflush(file_raw);

        /* RF Front End Simulation */
        for (int i=0;i<buf_size;i++){
            scaled = (float) buf[i] / maxval;
            sample_buf[i] = scaled * sin(phase) + scaled * cos(phase) * I;
            phase += step;
            if (phase >= max_phase)
                phase -= max_phase;
        }
        fwrite(sample_buf, sizeof(fcomplex), buf_size, file_iq);
        fflush(file_iq);

        /* Matched Filter */
        int len_filt_out_buf;
        memcpy(&filt_in_buf, &filt_in_buf[buf_size], mcs->rx_filter->num_taps*sizeof(fcomplex));
        memcpy(&filt_in_buf[mcs->rx_filter->num_taps-1], sample_buf, buf_size*sizeof(fcomplex));
        filt_out_buf = convolve_valid(filt_in_buf, filt_in_buf_size, mcs->rx_filter, &len_filt_out_buf);
        assert(buf_size == len_filt_out_buf);
        fwrite(filt_out_buf, sizeof(fcomplex), buf_size, file_flt);
        fflush(file_flt);

        /* Course Freq Sync */
        for (int i=0;i<buf_size;i++){
            freq_est_in_buf[i] = (double complex) cpow(filt_out_buf[i], order);
        }
        fftw_execute(plan);
        rewind(file_course_fft);
        fwrite(fft_buf, sizeof(double complex), buf_size, file_course_fft);
        fflush(file_course_fft);
        max = 0;
        for (int i=1;i<buf_size;i++){
           if (cabs(fft_buf[i]) > cabs(fft_buf[max]))
               max = i;
        }
        max = (max + buf_size/2)%buf_size; // fftshift
        freq_offset_est_hz = (-1*(float)mcs->input_sample_rate_hz/2) + ((float)mcs->input_sample_rate_hz / buf_size) * max;
        //freq_offset_est_hz = 0;
        printf("Frequency offset estimate: %f\n", freq_offset_est_hz/2);
        float course_adj_phase = 0;
        float t;
        for (int i=0;i<buf_size;i++){
            t = ((float) i)/((float) mcs->input_sample_rate_hz);
            filt_out_buf[i] = filt_out_buf[i] * exp(I*2*M_PI*(freq_offset_est_hz/2)*t);
        }
        fwrite(filt_out_buf, sizeof(fcomplex), buf_size, file_course);
        fflush(file_course);

        /* Time Sync */
        i_out = 2;
        int mu_log_idx = 0;
        while (i_out < buf_size && i_in+16 < buf_size){
            out[i_out] = filt_out_buf[i_in];
            real = 0;
            if (creal(out[i_out]) > 0)
                real = 1;
            imag = 0;
            if (cimag(out[i_out]) > 0)
                imag = 1;
            out_rail[i_out] = real + imag*I;
            x = (out_rail[i_out] - out_rail[i_out-2]) * conjf(out[i_out-1]);
            y = (out[i_out] - out[i_out-2]) * conjf(out_rail[i_out-1]);
            mm_val = creal(y - x);
            mu += ((float) samples_per_symbol) + mcs->mnm_aggression*mm_val;
            mnm_log[mu_log_idx++] = mcs->mnm_aggression*mm_val;
            i_in += (int) trunc(mu);
            mu = mu - trunc(mu);
            i_out += 1;
        }
        // save progress for next batch of samples
        out[0] = out[i_out-2];
        out[1] = out[i_out-1];
        out_rail[0] = out_rail[i_out-2];
        out_rail[1] = out_rail[i_out-1];
        i_in = i_in - buf_size;
        fcomplex *costas_in = &out[2];
        int len_samples = i_out-2;
        fwrite(mnm_log, sizeof(float), mu_log_idx, file_mnm_log);
        fflush(file_mnm_log);
        fwrite(&out[2], sizeof(fcomplex), i_out-2, file_mnm);
        fflush(file_mnm);


        /* Fine Frequency Sync */
        int N = len_samples;
        float alpha = 0.01;
        float beta = 0.000025;
        int freq_log_idx = 0;
        for(int i=0;i<N;i++){
            costas_out[i] = costas_in[i] * cexp(-1*I*costas_phase);
            error = creal(costas_out[i]) * cimag(costas_out[i]);
            freq_log[freq_log_idx++] = error;

            freq += (beta * error);
            freq_log[freq_log_idx++] = freq;
            costas_phase += freq + (alpha * error);
            freq_log[freq_log_idx++] = costas_phase;

            while (costas_phase >= 2*M_PI)
                costas_phase -= 2*M_PI;
            while (costas_phase < 0)
                costas_phase += 2*M_PI;
        }
        fwrite(freq_log, sizeof(float), freq_log_idx, file_ffs_ofst);
        fflush(file_ffs_ofst);
        fwrite(costas_out, sizeof(fcomplex), N, file_ffs);
        fflush(file_ffs);
        float max_amp = 0;

        /* Scale */
        for(int i=0;i<len_samples;i++){
            max_amp = fmax(max_amp, symbol_distance(costas_out[i], 0+0*I));
        }
        for(int i=0;i<len_samples;i++){
            costas_out[i] = costas_out[i] / max_amp;
        }
        fwrite(costas_out, sizeof(fcomplex), N, file_const);
        fflush(file_const);

        /* Demodulate */
        float min_dist, dist;
        int min_index;
        for(int i=0;i<len_samples;i++){
            min_dist = symbol_distance(costas_out[i], mcs->symbol_list_complex[0]);
            min_index = 0;
            for(int j=0;j<mcs->num_symbols;j++){
                dist = symbol_distance(costas_out[i], mcs->symbol_list_complex[j]);
                if(dist < min_dist)
                    min_index = j;
            }
            costas_out[i] = mcs->symbol_list_complex[min_index];
            int first_bit_index = i * mcs->bits_per_symbol;
            int first_byte_index = first_bit_index / 8;
            int last_bit_index = first_bit_index + mcs->bits_per_symbol;
            int last_byte_index = last_bit_index / 8;
            int shift = 8 - (first_bit_index % 8);
            if(first_byte_index == last_byte_index){
                bits[first_byte_index] |= mcs->symbol_list_int[min_index] << 8 - (first_bit_index % 8);
            }else if(first_byte_index == last_byte_index+1){
                bits[first_byte_index] |= (mcs->symbol_list_int[min_index] >> (1+last_bit_index % 8));
                bits[first_byte_index+1] |=  mcs->symbol_list_int[min_index] << 8 - (first_bit_index % 8);
            }
            //for(int b=first_byte_index; b<=last_byte_index; b++)
            //    bits[b] |= (mcs->symbol_list_int[min_index] << b & 0xf) << shift;
        }
        fwrite(costas_out, sizeof(fcomplex), len_samples, file_demod_sym);
        fflush(file_demod_sym);

        // frame detection

        // channel decode
    }
    fclose(file_raw);
    fclose(file_flt);
    fclose(file_iq);
    fclose(file_course_fft);
    fclose(file_course);
    fclose(file_mnm);
    fclose(file_mnm_log);
    fclose(file_ffs);
    fclose(file_ffs_ofst);

    fftw_destroy_plan(plan);
    fftw_cleanup();
    exit(0);


    return;
}
