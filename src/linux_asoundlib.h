#ifndef DOA_LINUX_ASOUNDLIB
#define DOA_LINUX_ASOUNDLIB
#include "mcs.h"
#include "circ_buf.h"

void start_rx_chain(MCS *mcs, CircBuf *sample_c_buf);

void start_tx_chain(MCS *mcs, CircBuf *iq_buf, CircBuf *sample_buf);

#endif
