#include "circ_buf.h"
#include <string.h>
#include <stdio.h>

/* Read len elem from circular buffer into out_buf.

returns the number of elements read.
*/
int read_buf(CircBuf *me, int nmemb, void *out_buf, int block){
    if(nmemb > me->count){
        if(block){
            nmemb = 0;
        }else{
            nmemb = me->count;
        }
    }

    char *typed_me = (char *) me->start;
    char *typed_out_buf = (char *) out_buf;
    for(int i=0;i<nmemb; i=i+1){
        memmove(&typed_out_buf[i*me->element_size],
                &typed_me[(i*me->element_size + me->read_idx*me->element_size) % (me->len*me->element_size)],
                me->element_size);
    }

    me->read_idx = (me->read_idx + nmemb) % me->len;
    me->count = me->count - nmemb;
    return nmemb;
}

/* Write nmemb elements from buffer into out_buf.

returns the number of elements written.
*/
int write_buf(void *in_buf, int nmemb, CircBuf *me, int block){
    int free = me->len - me->count;
    if(free < nmemb){
        if (block)
            return 0;
        else
            nmemb = free;
    }

    char *typed_me = (char *) me->start;
    char *typed_in_buf = (char *) in_buf;
    for(int i=0;i<nmemb;i=i+1){
        memmove(&typed_me[(me->write_idx*me->element_size + i*me->element_size)%(me->len*me->element_size)],
                &typed_in_buf[i*me->element_size],
                me->element_size);
    }
    if(me->stream != NULL){
        fwrite(in_buf, me->element_size, (size_t) nmemb, me->stream);
        fflush(me->stream);
    }

    me->write_idx = (me->write_idx + nmemb) % me->len;
    me->count = me->count + nmemb;
    return nmemb;
}
