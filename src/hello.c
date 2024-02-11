#include <emscripten/wasm_worker.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

struct circBuf {
    int *start; // pointer to start index
    int len; // total size
    int write_idx; // first free index
    int read_idx; // first used index
    int count; // number of read-able elements in the buf
};
typedef struct circBuf CircBuf ;

/* Read len bytes from circular buffer into out_buf.

returns the number of bytes read.
*/
int read_buf(CircBuf *in_buf, int len, int *out_buf){
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

int write_buf(int *in_buf, int len, CircBuf *out_buf){
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

void start_transmit(int in_buf){
    CircBuf *buf = (CircBuf *) in_buf;
    int tx_max_frame_size_bytes = 64;
    int* proc_buf = malloc(tx_max_frame_size_bytes);
    while(1){

        // read data from circular buffer
        int y = read_buf(buf, 1, proc_buf);

        // channel coding 
        // no channel coding for now
        
        // convert to symbols

        // do pulse shaping with matched filter. upscaling by samples per symbol

        // Do SDR part. Upconvert I & Q -> Sum -> send to Web Audio API
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
    int myArray[] = { 0xff, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12, 0x00, 0x11, 0x22, 0xff, 0xec, 0x12,0x00, 0x11, 0x22, 0xff, 0xec, 0x12};
 
    int input_buffer_len_bytes = 2048;
    int* tx_input_buffer = malloc(input_buffer_len_bytes);

    struct circBuf in_buf;
    in_buf.start = tx_input_buffer;
    in_buf.len = input_buffer_len_bytes;
    in_buf.read_idx = 0;
    in_buf.write_idx = 0;
    in_buf.count = 0;

    int x = write_buf(myArray, 24, &in_buf);

    emscripten_wasm_worker_t worker_tx = emscripten_malloc_wasm_worker(/*stackSize: */1024);
    emscripten_wasm_worker_post_function_vi(worker_tx, start_transmit,(int) &in_buf);

    emscripten_wasm_worker_t worker_rx = emscripten_malloc_wasm_worker(/*stackSize: */1024);
    emscripten_wasm_worker_post_function_v(worker_rx, start_receive);

    return 0;
};