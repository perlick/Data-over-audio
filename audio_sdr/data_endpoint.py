import multiprocessing as mp

class HexDataInput():
    def __init__(self, data_string, port_out):
        self.data_string = data_string
        self.port_out = port_out
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        if self.data_string is None:
            msg = 'deadbeef' # first 32 symbols are for time sync
            msg += '0123456789abcdeffefebaba'*10 # some random hex for fun
        by = bytes.fromhex(msg)

        for byte in by:
            self.port_out.put(byte)

class HexDataOutput():
    def __init__(self, port_in):
        """ port in is byte type
        """
        self.port_in = port_in

    def start(self):
        while True:
            byte = self.port_in.get()
            print(byte, end='')