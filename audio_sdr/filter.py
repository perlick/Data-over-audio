""" Components that act as filters
"""
import multiprocessing as mp

from scipy.signal import butter,filtfilt

class ButterLowpassFilter():
    def __init__(self, port_in, port_out, cutoff, sample_rate_hz, batch_size, order=2):
        """ butter lowpass filter has perfect response 
        
        cutoff: # desired cutoff frequency of the filter, Hz
        fs: sample rate of data
        order: Polynomial order of the signal. sin can be approximated as quadratic
        """
        self.port_in = port_in
        self.port_out = port_out
        self.cutoff = cutoff
        self.sample_rate_hz = sample_rate_hz
        self.batch_size = batch_size
        self.order = order
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        nyq = 0.5*self.sample_rate_hz
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        while True:
            data = self.port_in.get_buffered(self.batch_size)
            y = filtfilt(b, a, data)
            for s in y:
                self.port_out.put(s)