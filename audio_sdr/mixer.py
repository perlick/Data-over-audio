""" Components which are capable of combining or splitting signals
"""
import multiprocessing as mp


class Mixer():
    def __init__(self, port_in_1, port_in_2, port_out):
        self.port_in_1 = port_in_1
        self.port_in_2 = port_in_2
        self.port_out = port_out
        p = mp.Process(target=self.start)
        p.start()
    
    def start(self):
        while True:
            s_1 = self.port_in_1.get()
            s_2 = self.port_in_2.get()
            self.port_out.put(s_1 * s_2)

class Adder():
    def __init__(self, port_in_1, port_in_2, port_out):
        self.port_in_1 = port_in_1
        self.port_in_2 = port_in_2
        self.port_out = port_out
        p = mp.Process(target=self.start)
        p.start()
    
    def start(self):
        while True:
            s_1 = self.port_in_1.get()
            s_2 = self.port_in_2.get()
            self.port_out.put(s_1 + s_2)

class IQJoiner():
    def __init__(self, port_i_in, port_q_in, port_out):
        """
        port_i_in: queue of integers
        port_q_in: queue of integers
        port_out: queue of complex numbers
        """
        self.port_i_in = port_i_in
        self.port_q_in = port_q_in
        self.port_out = port_out
        p = mp.Process(target=self.start)
        p.start()
    
    def start(self):
        while True:
            s_i = self.port_i_in.get()
            s_q = self.port_q_in.get()
            self.port_out.put(s_i + s_q*1j)
