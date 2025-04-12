typedef struct {
  void (*start_tx_chain)(CircBuf *iq_buf, MCS *mcs);
  void (*start_rx_chain)(MCS *mcs);
} FrontEndInterface;