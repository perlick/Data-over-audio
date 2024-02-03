import numpy as np
import multiprocessing as mp

class Scaler():
    def __init__(self, port_in, port_out, batch_size):
        """
        port_in: integer samples
        port_out: scaled integer samples
        """
        self.port_in = port_in
        self.port_out = port_out
        self.batch_size = batch_size
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        while True:
            data = np.array(self.port_in.get_buffered(self.batch_size))
            data /= np.max(np.abs(data),axis=0)
            for s in data:
                self.port_out.put(s)
