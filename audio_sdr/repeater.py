import multiprocessing as mp

class Repeater():
    def __init__(self, port_in, port_out, symbol_freq_hz, sample_rate_hz):
        """
        port_in: queue of IQ symbols
        port_out: queue of IQ symbols
        samples_per_symbol: numer of times to repeat each symbol
            from the input port to the output port 
        """
        self.port_in = port_in
        self.port_out = port_out
        self.samples_per_symbol = sample_rate_hz / symbol_freq_hz
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        while True:
            sym = self.port_in.get()
            for _ in range(self.samples_per_symbol):
                self.port_out.put(sym)