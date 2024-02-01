import numpy as np
from scipy.io.wavfile import write

class WAVPlayer():
    def __init__(self, port_in, sample_rate_hz, filename):
        self.port_in = port_in
        self.sample_rate_hz = sample_rate_hz
        self.filename = filename

    def start(self):
        data = []
        try:
            while True:
                data.append(self.port_in.get(timeout=1))
        except:
            data = np.array(data)
            scaled = np.int16(data / np.max(np.abs(data)) * 32767)
            write(self.filename, int(self.sample_rate_hz), scaled)