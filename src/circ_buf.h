#ifndef DOA_CIRCBUF
#define DOA_CIRCBUF
#include <stddef.h> 
#include <stdio.h>

struct circBuf {
    void *start; // pointer to start index
    size_t element_size; // size of each element in bytes
    int len; // total size in elements
    int write_idx; // first free elem index
    int read_idx; // first used elem index
    int count; // number of read-able elements in the buf
    FILE *stream; // if set, stream all data that enters this buffer into FILE
};
typedef struct circBuf CircBuf;

int write_buf(void *in_buf, int len, CircBuf *me, int block);

int read_buf(CircBuf *me, int len, void *out_buf);

#endif
