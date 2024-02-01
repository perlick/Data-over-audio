class IQSplitter():
    def __init__(self, port_in, port_i_out, port_q_out):
        """
        Takes a single stream of complex numbers and splits into I and Q real numbers
        port_in: queue of single item complex numbers
        port_i_out: queue of single integer number from the real component of in
        port_q_out: queue of single integer number from the complex component of in
        """
        self.port_in = port_in
        self.port_i_out = port_i_out
        self.port_q_out = port_q_out

    def start(self):
        while True:
            s = self.port_in.get()
            self.port_i_out.put(s.real)
            self.port_q_out.put(s.imag)

class RFSplitter():
    def __init__(self, port_in, port_1_out, port_2_out):
        """
        Takes a single stream of integers and splits it into two identical streams
        port_in: queue of single integer numbers
        port_i_out: queue of single integer number
        port_q_out: queue of single integer number
        """
        self.port_in = port_in
        self.port_1_out = port_1_out
        self.port_2_out = port_2_out

    def start(self):
        while True:
            s = self.port_in.get()
            self.port_1_out.put(s)
            self.port_2_out.put(s)