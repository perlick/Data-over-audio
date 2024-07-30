#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <errno.h>
#include <getopt.h>
#include <alsa/asoundlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <complex.h>
#include <sys/mman.h> 
#include <signal.h>
#include <sys/wait.h>
#include <stdint.h>
#include <fftw3.h>
#include "circ_buf.h"
#include "filter.h"

#ifndef ESTRPIPE
#define ESTRPIPE ESPIPE
#endif

#define noop

static char *device = "default";         /* playback device */
static snd_pcm_format_t format = SND_PCM_FORMAT_S16;    /* sample format */
static unsigned int channels = 1;           /* count of channels */
static unsigned int rate;           /* stream rate */
static unsigned int buffer_time = 2e3;       /* ring buffer length in us */
static unsigned int period_time = 1e3;       /* period time in us */
static int method = 1;
static double freq;               /* sinusoidal wave frequency in Hz */
static int verbose = 0;                 /* verbose flag */
static int resample = 0;                /* enable alsa-lib resampling */
static int period_event = 1;                /* produce poll event after each period */
static int max_L2_packet_size_bytes = 1500;

static snd_pcm_sframes_t buffer_size;
static snd_pcm_sframes_t period_size;
static snd_output_t *output = NULL;

struct mcs {
    int channel_coding; // 0 is none
    int bits_per_symbol; // Number of bits per symbol
    int num_symbols; // Number of symbols / len of lists
    char *symbol_list_int; //Array of integer symbols. data representation
    float complex *symbol_list_complex; // Parallel array of complex numbers. signal representation
    int output_sample_rate_hz; // rate of the samples being sent to hardware device
    int symbol_rate_hz; // Rate of symbols being encoded into signal
    int carrier_freq_hz; // Tune Lo to this freq
    int input_sample_rate_hz; 
    int order; // Course frequency estimate multiplies by this number before taking FFT 
    float mnm_aggression;
    Filter *tx_filter;
    Filter *rx_filter;
};
typedef struct mcs MCS;

struct transfer_method {
    const char *name;
    snd_pcm_access_t access;
    int (*transfer_loop)(snd_pcm_t *handle,
                 signed short *samples,
                 snd_pcm_channel_area_t *areas,
                 CircBuf *iq_buf, int lo_freq);
};

int intPow(int x,int n)
{
    int i; /* Variable used in loop counter */
    int number = 1;

    for (i = 0; i < n; ++i)
        number *= x;

    return(number);
}

static void run_front_end_calculation(const snd_pcm_channel_area_t *areas, 
              snd_pcm_uframes_t offset,
              int count, double *_phase,
              CircBuf *iq_buf,
              int lo_freq, FILE *file)
{
    //printf("Front end: lo_freq(%d), rate(%d)\n", lo_freq, rate);
    static double max_phase = 2. * M_PI;
    double phase = *_phase;
    double step = max_phase*lo_freq/(double)rate;
    unsigned char *samples[channels];
    int steps[channels];
    unsigned int chn;
    int format_bits = snd_pcm_format_width(format);
    unsigned int maxval = (1 << (format_bits - 1)) - 1;
    int bps = format_bits / 8;  /* bytes per sample */
    int phys_bps = snd_pcm_format_physical_width(format) / 8;
    int big_endian = snd_pcm_format_big_endian(format) == 1;
    int to_unsigned = snd_pcm_format_unsigned(format) == 1;
    int is_float = (format == SND_PCM_FORMAT_FLOAT_LE ||
            format == SND_PCM_FORMAT_FLOAT_BE);
    float complex sample;
    int num_read;
 
    /* verify and prepare the contents of areas */
    for (chn = 0; chn < channels; chn++) {
        if ((areas[chn].first % 8) != 0) {
            printf("areas[%u].first == %u, aborting...\n", chn, areas[chn].first);
            exit(EXIT_FAILURE);
        }
        samples[chn] = /*(signed short *)*/(((unsigned char *)areas[chn].addr) + (areas[chn].first / 8));
        if ((areas[chn].step % 16) != 0) {
            printf("areas[%u].step == %u, aborting...\n", chn, areas[chn].step);
            exit(EXIT_FAILURE);
        }
        steps[chn] = areas[chn].step / 8;
        samples[chn] += offset * steps[chn];
    }

    /* fill the channel areas */
    while (count-- > 0) {
        // read a sample from the buffer
        num_read = read_buf(iq_buf, 1, &sample);
        // if there's nothing to read, play the carrier.
        if (num_read == 0)
            sample = 1 + 0*I; 

        union {
            float f;
            int i;
        } fval;
        short res, i;
        float inter;
        if (is_float) {
            fval.f = creal(sample) * sin(phase) + cimag(sample) * cos(phase);
            res = fval.i;
        } else {
            // Assumes amplitudes of I and Q do not exceed -1,1
            inter = creal(sample) * sin(phase) + cimag(sample) * cos(phase);
            inter = fmin(inter, 1);
            inter = fmax(inter, -1);
            res = inter * maxval;
            fwrite(&res, sizeof(short), 1, file);
            //printf("fe calc: I(%f) * sin(%f) + Q(%f) * cos(%f) * maxval(%d) = res(%d); \n", creal(sample), phase, cimag(sample), phase, maxval, res);
        }
        if (to_unsigned)
            res ^= 1U << (format_bits - 1);
        for (chn = 0; chn < channels; chn++) {
            /* Generate data in native endian format */
            if (big_endian) {
                for (i = 0; i < bps; i++)
                    *(samples[chn] + phys_bps - 1 - i) = (res >> i * 8) & 0xff;
            } else {
                for (i = 0; i < bps; i++)
                    *(samples[chn] + i) = (res >>  i * 8) & 0xff;
            }
            samples[chn] += steps[chn];
        }
        phase += step;
        if (phase >= max_phase)
            phase -= max_phase;
    }
    *_phase = phase;
}

/*
 *   Underrun and suspend recovery
 */
 
static int xrun_recovery(snd_pcm_t *handle, int err)
{
    if (verbose)
        printf("stream recovery\n");
    if (err == -EPIPE) {    /* under-run */
        err = snd_pcm_prepare(handle);
        if (err < 0)
            printf("Can't recovery from underrun, prepare failed: %s\n", snd_strerror(err));
        return 0;
    } else if (err == -ESTRPIPE) {
        while ((err = snd_pcm_resume(handle)) == -EAGAIN)
            sleep(1);   /* wait until the suspend flag is released */
        if (err < 0) {
            err = snd_pcm_prepare(handle);
            if (err < 0)
                printf("Can't recovery from suspend, prepare failed: %s\n", snd_strerror(err));
        }
        return 0;
    }
    return err;
}

static int set_hwparams(snd_pcm_t *handle,
            snd_pcm_hw_params_t *params,
            snd_pcm_access_t access,
            int unsigned rate)
{
    unsigned int rrate;
    snd_pcm_uframes_t size;
    int err, dir;
 
    /* choose all parameters */
    err = snd_pcm_hw_params_any(handle, params);
    if (err < 0) {
        printf("Broken configuration for playback: no configurations available: %s\n", snd_strerror(err));
        return err;
    }
    /* set hardware resampling */
    err = snd_pcm_hw_params_set_rate_resample(handle, params, resample);
    if (err < 0) {
        printf("Resampling setup failed for playback: %s\n", snd_strerror(err));
        return err;
    }
    /* set the interleaved read/write format */
    err = snd_pcm_hw_params_set_access(handle, params, access);
    if (err < 0) {
        printf("Access type not available for playback: %s\n", snd_strerror(err));
        return err;
    }
    /* set the sample format */
    err = snd_pcm_hw_params_set_format(handle, params, format);
    if (err < 0) {
        printf("Sample format not available for playback: %s\n", snd_strerror(err));
        return err;
    }
    /* set the count of channels */
    err = snd_pcm_hw_params_set_channels(handle, params, channels);
    if (err < 0) {
        printf("Channels count (%u) not available for playbacks: %s\n", channels, snd_strerror(err));
        return err;
    }
    /* set the stream rate */
    rrate = rate;
    err = snd_pcm_hw_params_set_rate_near(handle, params, &rrate, 0);
    if (err < 0) {
        printf("Rate %uHz not available for playback: %s\n", rate, snd_strerror(err));
        return err;
    }
    if (rrate != rate) {
        printf("Rate doesn't match (requested %uHz, get %iHz)\n", rate, err);
        return -EINVAL;
    }
    /* set the buffer time */
    err = snd_pcm_hw_params_set_buffer_time_near(handle, params, &buffer_time, &dir);
    if (err < 0) {
        printf("Unable to set buffer time %u for playback: %s\n", buffer_time, snd_strerror(err));
        return err;
    }
    err = snd_pcm_hw_params_get_buffer_size(params, &size);
    if (err < 0) {
        printf("Unable to get buffer size for playback: %s\n", snd_strerror(err));
        return err;
    }
    buffer_size = size;
    /* set the period time */
    err = snd_pcm_hw_params_set_period_time_near(handle, params, &period_time, &dir);
    if (err < 0) {
        printf("Unable to set period time %u for playback: %s\n", period_time, snd_strerror(err));
        return err;
    }
    err = snd_pcm_hw_params_get_period_size(params, &size, &dir);
    if (err < 0) {
        printf("Unable to get period size for playback: %s\n", snd_strerror(err));
        return err;
    }
    period_size = size;
    /* write the parameters to device */
    err = snd_pcm_hw_params(handle, params);
    if (err < 0) {
        printf("Unable to set hw params for playback: %s\n", snd_strerror(err));
        return err;
    }
    return 0;
}
 
static int set_swparams(snd_pcm_t *handle, snd_pcm_sw_params_t *swparams)
{
    int err;
 
    /* get the current swparams */
    err = snd_pcm_sw_params_current(handle, swparams);
    if (err < 0) {
        printf("Unable to determine current swparams for playback: %s\n", snd_strerror(err));
        return err;
    }
    /* start the transfer when the buffer is almost full: */
    /* (buffer_size / avail_min) * avail_min */
    err = snd_pcm_sw_params_set_start_threshold(handle, swparams, (buffer_size / period_size) * period_size);
    if (err < 0) {
        printf("Unable to set start threshold mode for playback: %s\n", snd_strerror(err));
        return err;
    }
    /* allow the transfer when at least period_size samples can be processed */
    /* or disable this mechanism when period event is enabled (aka interrupt like style processing) */
    err = snd_pcm_sw_params_set_avail_min(handle, swparams, period_event ? buffer_size : period_size);
    if (err < 0) {
        printf("Unable to set avail min for playback: %s\n", snd_strerror(err));
        return err;
    }
    /* enable period events when requested */
    if (period_event) {
        err = snd_pcm_sw_params_set_period_event(handle, swparams, 1);
        if (err < 0) {
            printf("Unable to set period event: %s\n", snd_strerror(err));
            return err;
        }
    }
    /* write the parameters to the playback device */
    err = snd_pcm_sw_params(handle, swparams);
    if (err < 0) {
        printf("Unable to set sw params for playback: %s\n", snd_strerror(err));
        return err;
    }
    return 0;
}

/*
 *   Transfer method - write and wait for room in buffer using poll
 */
 
static int wait_for_poll(snd_pcm_t *handle, struct pollfd *ufds, unsigned int count)
{
    unsigned short revents;
 
    while (1) {
        poll(ufds, count, -1);
        snd_pcm_poll_descriptors_revents(handle, ufds, count, &revents);
        if (revents & POLLERR)
            return -EIO;
        if (revents & POLLOUT)
            return 0;
    }
}
 
static int write_and_poll_loop(snd_pcm_t *handle,
                   signed short *samples,
                   snd_pcm_channel_area_t *areas,
                   CircBuf *iq_buf, int lo_freq)
{
    struct pollfd *ufds;
    double phase = 0;
    signed short *ptr;
    int err, count, cptr, init;
    FILE *plbk_raw = fopen("plbk_raw.s16", "w");
    //FILE *plbk_raw = NULL;
    
    count = snd_pcm_poll_descriptors_count (handle);
    if (count <= 0) {
        printf("Invalid poll descriptors count\n");
        return count;
    }
 
    ufds = malloc(sizeof(struct pollfd) * count);
    if (ufds == NULL) {
        printf("No enough memory\n");
        return -ENOMEM;
    }
    if ((err = snd_pcm_poll_descriptors(handle, ufds, count)) < 0) {
        printf("Unable to obtain poll descriptors for playback: %s\n", snd_strerror(err));
        return err;
    }
 
    init = 1;
    while (1) {
        if (!init) {
            err = wait_for_poll(handle, ufds, count);
            if (err < 0) {
                if (snd_pcm_state(handle) == SND_PCM_STATE_XRUN ||
                    snd_pcm_state(handle) == SND_PCM_STATE_SUSPENDED) {
                    err = snd_pcm_state(handle) == SND_PCM_STATE_XRUN ? -EPIPE : -ESTRPIPE;
                    if (xrun_recovery(handle, err) < 0) {
                        printf("Write error: %s\n", snd_strerror(err));
                        exit(EXIT_FAILURE);
                    }
                    init = 1;
                } else {
                    printf("Wait for poll failed\n");
                    return err;
                }
            }
        }
    
 
        run_front_end_calculation(areas, 0, period_size, &phase, iq_buf, lo_freq, plbk_raw);
        
        ptr = samples;
        cptr = period_size;
        while (cptr > 0) {
            err = snd_pcm_writei(handle, ptr, cptr);
            if (err < 0) {
                if (xrun_recovery(handle, err) < 0) {
                    printf("Write error: %s\n", snd_strerror(err));
                    exit(EXIT_FAILURE);
                }
                init = 1;
                break;  /* skip one period */
            }
            if (snd_pcm_state(handle) == SND_PCM_STATE_RUNNING)
                init = 0;
            ptr += err * channels;
            cptr -= err;
            if (cptr == 0)
                break;
            /* it is possible, that the initial buffer cannot store */
            /* all data from the last period, so wait awhile */
            err = wait_for_poll(handle, ufds, count);
            if (err < 0) {
                if (snd_pcm_state(handle) == SND_PCM_STATE_XRUN ||
                    snd_pcm_state(handle) == SND_PCM_STATE_SUSPENDED) {
                    err = snd_pcm_state(handle) == SND_PCM_STATE_XRUN ? -EPIPE : -ESTRPIPE;
                    if (xrun_recovery(handle, err) < 0) {
                        printf("Write error: %s\n", snd_strerror(err));
                        exit(EXIT_FAILURE);
                    }
                    init = 1;
                } else {
                    printf("Wait for poll failed\n");
                    return err;
                }
            }
        }
    }
}

static struct transfer_method transfer_methods[] = {
    { "write", SND_PCM_ACCESS_RW_INTERLEAVED, NULL },
    { "write_and_poll", SND_PCM_ACCESS_RW_INTERLEAVED, write_and_poll_loop },
    { "async", SND_PCM_ACCESS_RW_INTERLEAVED, NULL},
    { "async_direct", SND_PCM_ACCESS_MMAP_INTERLEAVED, NULL},
    { "direct_interleaved", SND_PCM_ACCESS_MMAP_INTERLEAVED, NULL},
    { "direct_noninterleaved", SND_PCM_ACCESS_MMAP_NONINTERLEAVED, NULL},
    { "direct_write", SND_PCM_ACCESS_MMAP_INTERLEAVED, NULL},
    { NULL, SND_PCM_ACCESS_RW_INTERLEAVED, NULL }
};


void start_tx_chain(CircBuf *iq_buf, MCS *mcs){
    snd_pcm_t *handle;
    int err, morehelp;
    rate = mcs->output_sample_rate_hz;
    freq = mcs->carrier_freq_hz;
    
    snd_pcm_hw_params_t *hwparams;
    snd_pcm_sw_params_t *swparams;
    
    signed short *samples;
    unsigned int chn;
    snd_pcm_channel_area_t *areas;
 
    snd_pcm_hw_params_alloca(&hwparams);
    snd_pcm_sw_params_alloca(&swparams);

    if (format == SND_PCM_FORMAT_LAST)
        format = SND_PCM_FORMAT_S16;
    if (!snd_pcm_format_linear(format) &&
        !(format == SND_PCM_FORMAT_FLOAT_LE ||
          format == SND_PCM_FORMAT_FLOAT_BE)) {
        printf("Invalid (non-linear/float) format %s\n",
               optarg);
        return;
    }

    err = snd_output_stdio_attach(&output, stdout, 0);
    if (err < 0) {
        printf("Output failed: %s\n", snd_strerror(err));
        return;
    }
    printf("Playback device is %s\n", device);
    printf("Stream parameters are %uHz, %s, %u channels\n", rate, snd_pcm_format_name(format), channels);
    printf("Using transfer method: %s\n", transfer_methods[method].name);
    fflush(stdout);
 
    if ((err = snd_pcm_open(&handle, device, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        printf("Playback open error: %s\n", snd_strerror(err));
        return;
    }
    
    if ((err = set_hwparams(handle, hwparams, transfer_methods[method].access, rate)) < 0) {
        printf("Setting of hwparams failed: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    if ((err = set_swparams(handle, swparams)) < 0) {
        printf("Setting of swparams failed: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    
    if (verbose > 0)
        snd_pcm_dump(handle, output);
    
    samples = malloc((period_size * channels * snd_pcm_format_physical_width(format)) / 8);
    if (samples == NULL) {
        printf("No enough memory\n");
        exit(EXIT_FAILURE);
    }
    
    areas = calloc(channels, sizeof(snd_pcm_channel_area_t));
    if (areas == NULL) {
        printf("No enough memory\n");
        exit(EXIT_FAILURE);
    }
    for (chn = 0; chn < channels; chn++) {
        areas[chn].addr = samples;
        areas[chn].first = chn * snd_pcm_format_physical_width(format);
        areas[chn].step = channels * snd_pcm_format_physical_width(format);
    }
 
    printf("starting transfer loop.\n");
    fflush(stdout);
 
    err = transfer_methods[method].transfer_loop(handle, samples, areas, iq_buf, mcs->carrier_freq_hz);
    if (err < 0)
        printf("Transfer failed: %s\n", snd_strerror(err));
 
    free(areas);
    free(samples);
    snd_pcm_close(handle);
    return;
}

void tx_encode_packet(CircBuf *buf, MCS *mcs, CircBuf *out_buf){
    // buffer for raw data
    char* data_buf = malloc(max_L2_packet_size_bytes);
    // buffer for symbols
    int symbol_buf_len = ((max_L2_packet_size_bytes*8) / mcs->bits_per_symbol) + 1;
    float complex *symbol_buf = malloc(symbol_buf_len);
    // buffer for upscaled symbols
    if (mcs->output_sample_rate_hz % mcs->symbol_rate_hz != 0){
    	printf("output_sample_rate_hz must be divisible by symbol_rate_hz!");
        exit(1);
    }
    int samples_per_symbol = mcs->input_sample_rate_hz / mcs->symbol_rate_hz;
    int sample_buf_len = symbol_buf_len * samples_per_symbol;
    float complex *sample_buf = malloc(sample_buf_len);

    // read data from circular buffer
    int num_read = read_buf(buf, max_L2_packet_size_bytes, data_buf);

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
        printf("symbol %d: %f + i%f\n", i, creal(symbol_buf[i]), cimag(symbol_buf[i]));
    }

    // do pulse shaping with matched filter. upscaling by samples per symbol
    // For now, do not do pulse shaping, just upscale 
    int sample_idx;
    int num_samples;
    num_samples = num_symbols * samples_per_symbol;
    for(int i=0;i<num_symbols;i++){
        sample_buf[i*samples_per_symbol] = symbol_buf[i];
    }
    FILE *plbk_sym = fopen("plbk_sym.fc32", "w");
    fwrite(sample_buf, sizeof(float complex), num_samples, plbk_sym);
    fclose(plbk_sym);

    int len_filt_buf;
    float complex *filt_buf = convolve(sample_buf, num_samples, mcs->tx_filter, &len_filt_buf);

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

void start_rx_chain(MCS *mcs){
    // read data from input buffer
    int i;
    int err;
    int buf_size = 2048;
    int16_t buf[buf_size];
    int rate = mcs->input_sample_rate_hz;
    int lo_freq = mcs->carrier_freq_hz;
    snd_pcm_t *capture_handle;
    snd_pcm_hw_params_t *hw_params;

    if ((err = snd_pcm_open(&capture_handle, device, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf (stderr, "cannot open audio device %s (%s)\n", 
             device, 
             snd_strerror(err));
        exit(1);
    }
       
    if ((err = snd_pcm_hw_params_malloc (&hw_params)) < 0) {
        fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n",
             snd_strerror(err));
        exit(1);
    }
             
    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0) {
        fprintf (stderr, "cannot initialize hardware parameter structure (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf (stderr, "cannot set access type (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, format)) < 0) {
        fprintf (stderr, "cannot set sample format (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0)) < 0) {
        fprintf (stderr, "cannot set sample rate (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 1)) < 0) {
        fprintf (stderr, "cannot set channel count (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0) {
        fprintf (stderr, "cannot set parameters (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    snd_pcm_hw_params_free(hw_params);

    if ((err = snd_pcm_prepare(capture_handle)) < 0) {
        fprintf (stderr, "cannot prepare audio interface for use (%s)\n",
             snd_strerror(err));
        exit(1);
    }

    FILE *file_raw = fopen("cap_raw.s16", "w");
    FILE *file_flt = fopen("cap_flt.fc32", "w");
    FILE *file_iq = fopen("cap_iq.fc32", "w");
    FILE *file_course_fft = fopen("cap_course.fft", "w");
    FILE *file_course = fopen("cap_course.fc32", "w");
    FILE *file_mnm = fopen("cap_mnm.fc32", "w");
    //printf("Front end: lo_freq(%d), rate(%d)\n", lo_freq, rate);
    static double max_phase = 2. * M_PI;
    double phase = 0;
    double step = max_phase*lo_freq/(double)rate;
    int format_bits = snd_pcm_format_width(format);
    unsigned int maxval = (1 << (format_bits - 1)) - 1;
    int bps = format_bits / 8;  /* bytes per sample */
    int phys_bps = snd_pcm_format_physical_width(format) / 8;
    int big_endian = snd_pcm_format_big_endian(format) == 1;
    int to_unsigned = snd_pcm_format_unsigned(format) == 1;
    int is_float = (format == SND_PCM_FORMAT_FLOAT_LE ||
            format == SND_PCM_FORMAT_FLOAT_BE);
    float complex sample_buf[buf_size];
    int filt_in_buf_size = buf_size + mcs->rx_filter->num_taps - 1;
    float complex filt_in_buf[filt_in_buf_size];
    memset(filt_in_buf, 0, sizeof(float complex)*filt_in_buf_size);
    float complex *filt_out_buf;
    double complex freq_est_in_buf[buf_size];
    float scaled;
    int order = mcs->order;
    fftw_complex fft_buf[buf_size];
    fftw_plan plan = fftw_plan_dft_1d(buf_size, freq_est_in_buf, fft_buf, FFTW_FORWARD, FFTW_MEASURE);
    int max;
    float freq_offset_est_hz;
    float mu = 0;
    float complex out[buf_size + 10];
    float complex out_rail[buf_size + 10];
    int i_in, i_out;
    float mm_val, real, imag;
    float complex x, y;
    int samples_per_symbol = mcs->output_sample_rate_hz / mcs->symbol_rate_hz;
    while (1) {
        /* Get Raw Samples */
        if ((err = snd_pcm_readi(capture_handle, buf, buf_size)) != buf_size) {
            fprintf (stderr, "read from audio interface failed (%s)\n",
                 snd_strerror(err));
            exit(1);
        }
        fwrite(buf, sizeof(int16_t), buf_size, file_raw);
 
        /* RF Front End Simulation */
        for (int i=0;i<buf_size;i++){
            scaled = (float) buf[i] / maxval; 
            sample_buf[i] = scaled * sin(phase) + scaled * cos(phase) * I;
            phase += step;
            if (phase >= max_phase)
                phase -= max_phase;
        }
        fwrite(sample_buf, sizeof(float complex), buf_size, file_iq);

        /* Matched Filter */
        int len_filt_out_buf;
        memcpy(&filt_in_buf, &filt_in_buf[buf_size], mcs->rx_filter->num_taps*sizeof(float complex));
        memcpy(&filt_in_buf[mcs->rx_filter->num_taps-1], sample_buf, buf_size*sizeof(float complex));
        filt_out_buf = convolve_valid(filt_in_buf, filt_in_buf_size, mcs->rx_filter, &len_filt_out_buf);
        assert(buf_size == len_filt_out_buf);
        fwrite(filt_out_buf, sizeof(float complex), buf_size, file_flt);

        /* Course Freq Sync */
        for (int i=0;i<buf_size;i++){
            freq_est_in_buf[i] = (double complex) cpow(filt_out_buf[i], order); 
        }
        fftw_execute(plan);
        rewind(file_course_fft);
        fwrite(fft_buf, sizeof(double complex), buf_size, file_course_fft);
        max = 0;
        for (int i=1;i<buf_size;i++){
           if (cabs(fft_buf[i]) > cabs(fft_buf[max]))
               max = i;
        }
        max = (max + buf_size/2)%buf_size; // fftshift
        freq_offset_est_hz = (-1*(float)mcs->input_sample_rate_hz/2) + ((float)mcs->input_sample_rate_hz / buf_size) * max; 
        printf("Frequency offset estimate: %f\n", freq_offset_est_hz/2);
        float course_adj_phase = 0;
        float t;
        for (int i=0;i<buf_size;i++){
            t = i/mcs->input_sample_rate_hz;
            sample_buf[i] = sample_buf[i] * exp(-1*I*2*M_PI*freq_offset_est_hz*t/2);
        }
        fwrite(sample_buf, sizeof(float complex), buf_size, file_course);

        /* Time Sync */
        i_in = 0;
        i_out = 2; 
        while (i_out < buf_size && i_in+16 < buf_size){
            out[i_out] = sample_buf[i_in + (int)mu];
            real = 0;
            if (creal(out[i_out]) > 0)
                real = 1;
            imag = 0;
            if (cimag(out[i_out]) > 0)
                imag = 1;
            out_rail[i_out] = real + imag*I;
            x = (out_rail[i_out] - out_rail[i_out-2]) * conj(out[i_out-1]);
            y = (out[i_out] - out[i_out-2]) * conj(out_rail[i_out-1]);
            mm_val = creal(y - x);
            mu += samples_per_symbol + mcs->mnm_aggression*mm_val;
            i_in += (int) trunc(mu);
            mu = mu - trunc(mu);
            i_out += 1;
        }
        fwrite(&out[2], sizeof(float complex), i_out-2, file_mnm);

        // fine freq sync

        // demodulate

        // frame detection

        // channel decode
    }
    fclose(file_raw);
    fclose(file_flt);
    fclose(file_iq);
    fclose(file_course_fft);
    fclose(file_course);
    fclose(file_mnm);

    fftw_destroy_plan(plan);
    fftw_cleanup();
    snd_pcm_close (capture_handle);
    exit(0);


    return;
}

void* create_shared_memory(size_t size){
    int protection = PROT_READ | PROT_WRITE;
    int visibility = MAP_SHARED | MAP_ANONYMOUS;
    return mmap(NULL, size, protection, visibility, -1, -0);

}

static void handler(int signum){
    while(wait(NULL) != -1 || errno == EINTR);
    exit(1);
}

int main(){
    /* setup bsaic handlers*/
    struct sigaction sa;
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGINT, &sa, NULL) == -1){
        printf("failed to register signal handler SIGINT: %s\n", strerror(errno));
        exit(1);
    }

    char myArray[] = { 0xff, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12,0x00, 0x11, 0x22, 0xff, 0xec, 0x12};
 
    char* tx_input_buffer = malloc(max_L2_packet_size_bytes);

    struct circBuf in_buf;
    in_buf.element_size = sizeof(char);
    in_buf.start = tx_input_buffer;
    in_buf.len = max_L2_packet_size_bytes;
    in_buf.read_idx = 0;
    in_buf.write_idx = 0;
    in_buf.count = 0;

    int x = write_buf(myArray, 24, &in_buf, 1);
 
    /* bpsk */
    struct mcs mcs0;
    mcs0.channel_coding = 0;
    mcs0.bits_per_symbol = 1;
    mcs0.num_symbols = 2;
    mcs0.symbol_list_int = malloc(mcs0.num_symbols);
    mcs0.symbol_list_complex = malloc(mcs0.num_symbols * sizeof(complex float));
    mcs0.symbol_list_int[0] = 0;
    mcs0.symbol_list_complex[0] = 1 + 0*I;
    mcs0.symbol_list_int[1] = 1;
    mcs0.symbol_list_complex[1] = -1 + 0*I;
    mcs0.output_sample_rate_hz = 8000;
    mcs0.symbol_rate_hz = 100;
    mcs0.carrier_freq_hz = 440;
    mcs0.input_sample_rate_hz = 8000;
    mcs0.order = 2;
    mcs0.mnm_aggression = 0.3;
    mcs0.tx_filter = create_filter_rrc1(12.625, 0.35, (float) mcs0.output_sample_rate_hz / mcs0.symbol_rate_hz);
    mcs0.rx_filter = create_filter_rrc1(12.625, 0.35, (float) mcs0.input_sample_rate_hz / mcs0.symbol_rate_hz);

    struct mcs mcs1;
    mcs1.channel_coding = 0;
    mcs1.bits_per_symbol = 2;
    mcs1.num_symbols = 4;
    mcs1.symbol_list_int = malloc(mcs1.num_symbols);
    mcs1.symbol_list_complex = malloc(mcs1.num_symbols * sizeof(complex float));
    mcs1.symbol_list_int[0] = 0;
    mcs1.symbol_list_complex[0] = 1 + 0*I;
    mcs1.symbol_list_int[1] = 1;
    mcs1.symbol_list_complex[1] = -1 + 0*I;
    mcs1.symbol_list_int[2] = 2;
    mcs1.symbol_list_complex[2] = 0 + 1*I;
    mcs1.symbol_list_int[3] = 3;
    mcs1.symbol_list_complex[3] = 0 + -1*I;
    mcs1.output_sample_rate_hz = 8000;
    mcs1.symbol_rate_hz = 100;
    mcs1.carrier_freq_hz = 440;
    mcs1.input_sample_rate_hz = 8000;
    mcs1.order = 4;
    mcs1.mnm_aggression = 0.3;
    mcs1.tx_filter = create_filter_rrc1(12.625, 0.35, (float) mcs1.output_sample_rate_hz / mcs1.symbol_rate_hz);
    mcs1.rx_filter = create_filter_rrc1(12.625, 0.35, (float) mcs1.input_sample_rate_hz / mcs1.symbol_rate_hz);

    struct mcs *cur_mcs = &mcs0;

    // put the fe buffer in a shared memory space. 
    int samples_per_symbol = cur_mcs->output_sample_rate_hz / cur_mcs->symbol_rate_hz;
    int max_L2_packet_size_samples = (max_L2_packet_size_bytes * 8 / cur_mcs->bits_per_symbol) * samples_per_symbol + cur_mcs->tx_filter->num_taps + 1; 
    char* fe_input_buffer = create_shared_memory(max_L2_packet_size_samples * sizeof(float complex));
    struct circBuf *fe_buf = create_shared_memory(sizeof(struct circBuf));
    fe_buf->element_size = sizeof(float complex);
    fe_buf->start = fe_input_buffer;
    fe_buf->len = max_L2_packet_size_samples;
    fe_buf->read_idx = 0;
    fe_buf->write_idx = 0;
    fe_buf->count = 0;
    fe_buf->stream = fopen("plbk_iq.fc32", "w");
    //fe_buf->stream = NULL;

    /* start listening */
    pid_t rx_fe_child;
    if ((rx_fe_child=fork())==0){
        start_rx_chain(cur_mcs);
    }

    // write a packet of IQ samples to buffer
    tx_encode_packet(&in_buf, cur_mcs, fe_buf);
    
    pid_t tx_fe_child;
    if ((tx_fe_child=fork())==0){
        start_tx_chain(fe_buf, cur_mcs);
    }
 

    char *line = NULL;
    size_t size;
    size_t num;
    while (1){
        num = getline(&line, &size, stdin);
        if (num == -1) {
            printf("No line\n");
        } else {
            x = write_buf(line, num, &in_buf, 1);
            tx_encode_packet(&in_buf, &mcs1, fe_buf);
        }
    }

    /* send user inputs! */
    while(wait(NULL) != -1 || errno == EINTR){
    
    }
};

