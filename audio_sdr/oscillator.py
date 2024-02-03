import multiprocessing as mp
import math

class LocalOscillator():
    def __init__(self, port_out, tune_freq_hz, sample_rate_hz, func):
        self.port_out = port_out
        self.tune_freq_hz = tune_freq_hz
        self.sample_rate_hz = sample_rate_hz
        if func == "sin":
            self.func = math.sin
        if func == "cos":
            self.func = math.cos
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        t=0
        while True:
            lo_s = self.func(2*math.pi*self.tune_freq_hz*t)
            self.port_out.put(lo_s)
            t = t + 1/self.sample_rate_hz