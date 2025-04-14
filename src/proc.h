#ifndef DOA_PROC
#define DOA_PROC
#include "mcs.h"
#include "circ_buf.h"

void spawn_rx_chain(MCS *cur_mcs, CircBuf *sample_buf);

void spawn_tx_chain(MCS *cur_mcs, CircBuf *fe_buf, CircBuf *sample_buf);

#endif
