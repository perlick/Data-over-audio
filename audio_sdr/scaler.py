import numpy as np

class Scaler():
    def __init__(self, port_in, port_out, batch_size):
        """
        port_in: integer samples
        port_out: scaled integer samples
        """
        self.port_in = port_in
        self.port_out = port_out
        self.batch_size = batch_size

    def start(self):
        while True:
            data = np.array(self.port_in.get_buffered(self.batch_size))
            data /= np.max(np.abs(data),axis=0)
            for s in data:
                self.port_out.put(s)
