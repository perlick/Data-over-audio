#include "proc.h"
#include "circ_buf.h"
#include "mcs.h"
#ifdef _WIN32
#include <windows.h>
void spawn_rx_chain(){
    /* start listening */
    pid_t rx_fe_child;
    if ((rx_fe_child=fork())==0){
        start_rx_chain(cur_mcs);
    }
}

void spawn_tx_chain(){
    pid_t tx_fe_child;
    if ((tx_fe_child=fork())==0){
        start_tx_chain(fe_buf, cur_mcs);
    }
}

#elif __linux
#include <sys/types.h>
#include "linux_asoundlib.h"
#include <unistd.h>
void spawn_rx_chain(MCS *cur_mcs, CircBuf *sample_buf){
    /* start listening */
    pid_t rx_fe_child;
    if ((rx_fe_child=fork())==0){
        start_rx_chain(cur_mcs, sample_buf);
    }
}

void spawn_tx_chain(MCS *cur_mcs, CircBuf *fe_buf, CircBuf *sample_buf){
    pid_t tx_fe_child;
    if ((tx_fe_child=fork())==0){
        start_tx_chain(cur_mcs, fe_buf, sample_buf);
    }
}

#endif
