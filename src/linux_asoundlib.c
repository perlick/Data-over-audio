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

struct transfer_method {
    const char *name;
    snd_pcm_access_t access;
    int (*transfer_loop)(snd_pcm_t *handle,
                 signed short *samples,
                 snd_pcm_channel_area_t *areas,
                 CircBuf *iq_buf, int lo_freq);
};

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

static void run_front_end_calculation(const snd_pcm_channel_area_t *areas, 
              snd_pcm_uframes_t offset,
              int count, double *_phase,
              CircBuf *iq_buf,
              int lo_freq, FILE *file){
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
    fcomplex sample;
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
 
static int xrun_recovery(snd_pcm_t *handle, int err){
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
            int unsigned rate){
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


static int set_swparams(snd_pcm_t *handle, snd_pcm_sw_params_t *swparams){
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
 
static int wait_for_poll(snd_pcm_t *handle, struct pollfd *ufds, unsigned int count){
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
                   CircBuf *iq_buf, int lo_freq){
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
    FILE *file_mnm_log = fopen("cap_mnm_log.fc32", "w");
    FILE *file_ffs = fopen("cap_ffs.fc32", "w");
    FILE *file_ffs_ofst = fopen("cap_ffs_log.f3c32", "w");
    FILE *file_const = fopen("cap_const.const", "w");
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
    int i_in, i_out;
    float mm_val, real, imag;
    fcomplex x, y;
    float mnm_log[buf_size*2];
    int samples_per_symbol = mcs->input_sample_rate_hz / mcs->symbol_rate_hz;
    float costas_phase = 0;
    float freq = 0;
    float error;
    fcomplex costas_out[buf_size];
    float freq_log[buf_size*2];
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
        fwrite(sample_buf, sizeof(fcomplex), buf_size, file_iq);

        /* Matched Filter */
        int len_filt_out_buf;
        memcpy(&filt_in_buf, &filt_in_buf[buf_size], mcs->rx_filter->num_taps*sizeof(fcomplex));
        memcpy(&filt_in_buf[mcs->rx_filter->num_taps-1], sample_buf, buf_size*sizeof(fcomplex));
        filt_out_buf = convolve_valid(filt_in_buf, filt_in_buf_size, mcs->rx_filter, &len_filt_out_buf);
        assert(buf_size == len_filt_out_buf);
        fwrite(filt_out_buf, sizeof(fcomplex), buf_size, file_flt);

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
            t = ((float) i)/((float) mcs->input_sample_rate_hz);
            filt_out_buf[i] = filt_out_buf[i] * exp(I*2*M_PI*(freq_offset_est_hz/2)*t);
        }
        fwrite(filt_out_buf, sizeof(fcomplex), buf_size, file_course);

        /* Time Sync */
        i_in = 0;
        i_out = 2; 
        int mu_log_idx = 0;
        while (i_out < buf_size && i_in+16 < buf_size){
            out[i_out] = filt_out_buf[i_in + (int)mu];
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
            mnm_log[mu_log_idx++] = mm_val;
            mnm_log[mu_log_idx++] = mu;
            i_in += (int) trunc(mu);
            mu = mu - trunc(mu);
            i_out += 1;
        }
        fcomplex *costas_in = &out[2];
        int len_samples = i_out-2;
        fwrite(mnm_log, sizeof(float), mu_log_idx, file_mnm_log);
        fwrite(&out[2], sizeof(fcomplex), i_out-2, file_mnm);

        /* Fine Frequency Sync */
        int N = len_samples;
        float alpha = 0.132;
        float beta = 0.00932;
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
        fwrite(costas_out, sizeof(fcomplex), N, file_ffs);
        fwrite(costas_out, sizeof(fcomplex), N, file_const);

        /* Demodulate */
        
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
    snd_pcm_close (capture_handle);
    exit(0);


    return;
}