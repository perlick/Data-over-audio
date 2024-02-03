import numpy as np
from scipy.io.wavfile import read
import multiprocessing as mp

class WAVSampler():
    def __init__(self, port_out, filename):
        self.port_out = port_out
        self.filename = filename
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        a = read(self.filename)
        sample_rate_hz = a[0]
        data = np.array(a[1], dtype=float)
        for i in data:
            self.port_out.put(i)