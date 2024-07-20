#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <complex.h>

#define noop

struct circBuf {
    char *start; // pointer to start index
    int len; // total size
    int write_idx; // first free index
    int read_idx; // first used index
    int count; // number of read-able elements in the buf
};
typedef struct circBuf CircBuf ;

/* Read len bytes from circular buffer into out_buf.

returns the number of bytes read.
*/
int read_buf(CircBuf *in_buf, int len, char *out_buf){
    // check if there's enough bytes to read
    if(len > in_buf->count){
        len = in_buf->count;
    }
    // do the reading
    for(int i=0;i<len; i=i+1){
        out_buf[i] = in_buf->start[(i+in_buf->read_idx)%in_buf->len];
    }
    // move the read index
    in_buf->read_idx = (in_buf->read_idx + len) % in_buf->len;
    // decrement the count
    in_buf->count = in_buf->count - len;
    return len;
}

int write_buf(char *in_buf, int len, CircBuf *out_buf){
    // check how many elements can be written
    int free = out_buf->len - out_buf->count;
    // set len to be written
    if(free < len){
        len = free;
    }
    // do the writing
    for(int i=0;i<len;i=i+1){
        out_buf->start[(out_buf->write_idx + i)%out_buf->len] = in_buf[i];
    }
    // move the write index
    out_buf->write_idx = (out_buf->write_idx + len) % out_buf->len;
    // update count
    out_buf->count = out_buf->count + len;
    printf("buf_size: %d \n", out_buf->count);
    return len;
}

int intPow(int x,int n)
{
    int i; /* Variable used in loop counter */
    int number = 1;

    for (i = 0; i < n; ++i)
        number *= x;

    return(number);
}

struct mcs {
    int channel_coding; // 0 is none
    int bits_per_symbol; // number of bits per symbol
    int num_symbols; // number of symbols / len of lists
    char *symbol_list_int; //array of integer symbols
    float complex *symbol_list_complex; // parallel array of complex numbers

};
typedef struct mcs MCS;

void start_transmit(CircBuf *buf, MCS *mcs){
    int tx_max_frame_size_bytes = 64;
    char* data_buf = malloc(tx_max_frame_size_bytes);
    int symbol_buf_len = ((tx_max_frame_size_bytes*8) / mcs->bits_per_symbol) + 1;
    float complex *symbol_buf = malloc(symbol_buf_len);
    while(1){

        // read data from circular buffer
        int num_read = read_buf(buf, tx_max_frame_size_bytes, data_buf);

        // channel coding 
        if (mcs->channel_coding==0){
            noop;
        }
        
        // convert to symbols
        int num_symbols = (((num_read*8) + mcs->bits_per_symbol - 1) / mcs->bits_per_symbol);
        printf("num symbols: %d\n", num_symbols);
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
            for(int j=0;j<mcs->num_symbols;j=j+1){
                if(sym_int == mcs->symbol_list_int[j]){
                    symbol_buf[i] = mcs->symbol_list_complex[j];
                    break;
                }
            }
            printf("symbol %d: %f + i%f\n", i, creal(symbol_buf[i]), cimag(symbol_buf[i]));
        }

        // do pulse shaping with matched filter. upscaling by samples per symbol

        // Do SDR part. Upconvert I & Q -> Sum -> send to Web Audio API

        sleep(1);
    }
    return;
}

void start_receive(){
    // read data from input buffer

    // do SDR part. split -> Downconvert

    // matched filter

    // course freq sync

    // time sync (decimate)

    // fine freq sync

    // demodulate

    // frame detection

    // channel decode

    return;
}

int main(){
    char myArray[] = { 0xff, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12,0x00, 0x11, 0x22, 0xff, 0xec, 0x12};
 
    int input_buffer_len_bytes = 2048;
    char* tx_input_buffer = malloc(input_buffer_len_bytes);

    struct circBuf in_buf;
    in_buf.start = tx_input_buffer;
    in_buf.len = input_buffer_len_bytes;
    in_buf.read_idx = 0;
    in_buf.write_idx = 0;
    in_buf.count = 0;

    int x = write_buf(myArray, 24, &in_buf);

    struct mcs mcs1;
    mcs1.channel_coding = 0;
    mcs1.bits_per_symbol = 2;
    mcs1.num_symbols = 4;
    mcs1.symbol_list_int = malloc(mcs1.num_symbols);
    mcs1.symbol_list_complex = malloc(mcs1.num_symbols * sizeof(*mcs1.symbol_list_complex));
    mcs1.symbol_list_int[0] = 0;
    mcs1.symbol_list_complex[0] = 1 + I;
    mcs1.symbol_list_int[1] = 1;
    mcs1.symbol_list_complex[1] = 1 + -1*I;
    mcs1.symbol_list_int[2] = 2;
    mcs1.symbol_list_complex[2] = -1 + I;
    mcs1.symbol_list_int[3] = 3;
    mcs1.symbol_list_complex[3] = -1 + -1*I;
    

    start_transmit(&in_buf, &mcs1);

    return 0;
};
