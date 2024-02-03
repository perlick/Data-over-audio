import multiprocessing as mp

class UnalignedDecimator():
    def __init__(self, port_in, port_out, symbol_rate_hz, sample_rate_hz):
        self.port_in = port_in
        self.port_out = port_out
        self.symbol_rate_hz = symbol_rate_hz
        self.sample_rate_hz = sample_rate_hz
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        samples_per_symbol = self.sample_rate_hz / self.symbol_rate_hz
        while True:
            samples = self.port_in.get_buffered(samples_per_symbol)
            self.port_out.put(samples[0])