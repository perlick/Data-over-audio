#include "circ_buf.h"
#include <string.h>
#include <stdio.h>

/* Read len elem from circular buffer into out_buf.

returns the number of bytes read.
*/
int read_buf(CircBuf *me, int len, void *out_buf){
    if(len > me->count){
        len = me->count;
    }

    char *typed_me = (char *) me->start;
    char *typed_out_buf = (char *) out_buf;
    for(int i=0;i<len; i=i+1){
        memmove(&typed_out_buf[i*me->element_size],
                &typed_me[(i*me->element_size + me->read_idx*me->element_size) % (me->len*me->element_size)],
                me->element_size);
    }

    me->read_idx = (me->read_idx + len) % me->len;
    me->count = me->count - len;
    return len;
}

/* Write len bytes from buffer into out_buf.

returns the number of bytes written.
*/
int write_buf(void *in_buf, int len, CircBuf *me, int block){
    int free = me->len - me->count;
    if(free < len){
        if (block)
            return 0;
        else
            len = free;
    }

    char *typed_me = (char *) me->start;
    char *typed_in_buf = (char *) in_buf;
    for(int i=0;i<len;i=i+1){
        memmove(&typed_me[(me->write_idx*me->element_size + i*me->element_size)%(me->len*me->element_size)],
                &typed_in_buf[i*me->element_size],
                me->element_size);
    }
    if(me->stream != NULL){
        fwrite(in_buf, me->element_size, (size_t) len, me->stream);
    }
    
    me->write_idx = (me->write_idx + len) % me->len;
    me->count = me->count + len;
    return len;
}
