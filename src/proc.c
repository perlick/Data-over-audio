#ifdef _WIN32
#include <windows.h>
void spwan_rx_chain(){
    /* start listening */
    pid_t rx_fe_child;
    if ((rx_fe_child=fork())==0){
        start_rx_chain(cur_mcs);
    }
}

void spwan_tx_chain(){
    pid_t tx_fe_child;
    if ((tx_fe_child=fork())==0){
        start_tx_chain(fe_buf, cur_mcs);
    }
}

#elif __linux

void spwan_rx_chain(){
    /* start listening */
    pid_t rx_fe_child;
    if ((rx_fe_child=fork())==0){
        start_rx_chain(cur_mcs);
    }
}

void spwan_tx_chain(){
    pid_t tx_fe_child;
    if ((tx_fe_child=fork())==0){
        start_tx_chain(fe_buf, cur_mcs);
    }
}

#endif