import logging
import math
import numpy as np
import multiprocessing as mp

logger = logging.getLogger()

QAM_SYMBOLS_ENCODE = {
    "bit_len": 2,
    "symbol_map": {
        0b00: -1 + -1j,
        0b01: -1 + 1j,
        0b10: 1 + -1j,
        0b11: 1 + 1j
        }
    }
QAM_SYMBOLS_DECODE = {
    "bit_len": 2,
    "symbol_map": {}
}
for key, val in QAM_SYMBOLS_ENCODE["symbol_map"].copy().items():
    QAM_SYMBOLS_DECODE['symbol_map'][val] = key

class Symbolizer():
    encodings = {
        "QAM": QAM_SYMBOLS_ENCODE
    }
    def __init__(self, port_in, port_out, encoding):
        """
        port_in: expects single byte items in queue
        port_out: complex number items
        encoding: QAM
        """
        if encoding not in Symbolizer.encodings.keys():
            logger.critical(f"no such encoding: {encoding}")
            raise NameError

        self.port_in = port_in
        self.port_out = port_out
        self.symbol_map = Symbolizer.encodings[encoding]['symbol_map']
        self.symbol_len = Symbolizer.encodings[encoding]['bit_len']
        p = mp.Process(target=self.start)
        p.start()

    def start(self):
        data = bytearray()
        remainder_offset = 0
        num_data_bits = len(data) * 8 - remainder_offset
        while True:
            print("here")
            while num_data_bits < self.symbol_len:
                data.append(self.port_in.get())
                num_data_bits = len(data) * 8 - remainder_offset
            print(f"symbolizer found data! {data}")
            mask = 2**self.symbol_len - 1
            num_symbols = int(num_data_bits // self.symbol_len)
            remainder = num_data_bits % self.symbol_len
            remainder_offset = 8 - (remainder % 8)
            symbol_vec = np.zeros(num_symbols, dtype=np.complex_)
            for i in range(num_symbols):
                first_bit_index = i * self.symbol_len + remainder_offset
                first_byte_index = first_bit_index // 8
                last_bit_intex = first_bit_index + self.symbol_len + remainder_offset
                last_byte_index = last_bit_intex // 8 
                value = int.from_bytes(data[first_byte_index:last_byte_index+1], byteorder='little')
                shift = first_bit_index % 8
                symbol_vec[i] = self.symbol_map[value >> shift & mask]
            data = data[last_byte_index:]
            for s in symbol_vec:
                self.port_out.put(s)
            
def iq_distance(symbol_1, symbol_2):
    i_1 = symbol_1.real
    i_2 = symbol_2.real
    q_1 = symbol_1.imag
    q_2 = symbol_2.imag

    return math.sqrt((i_1 - i_2)**2 + (q_1 - q_2)**2)

class DeSymbolizer():
    encodings = {
        "QAM": QAM_SYMBOLS_DECODE
    }
    def __init__(self, port_in, port_out, encoding):
        self.port_in = port_in
        self.port_out = port_out
        self.symbol_map = DeSymbolizer.encodings[encoding]['symbol_map']
        self.symbol_len = DeSymbolizer.encodings[encoding]['bit_len']
        p = mp.Process(target=self.start)
        p.start()
    
    def start(self):
        decoded_bits = 0
        buffer = 0
        while True:
            symbol = self.port_in.get()
            min_distance = 1000
            best_symbol = None
            for ref_symbol in self.symbol_map.keys():
                if best_symbol is None:
                    best_symbol = ref_symbol
                    min_distance = iq_distance(ref_symbol, symbol)
                    continue
                dist = iq_distance(ref_symbol, symbol)
                if dist < min_distance:
                    best_symbol = ref_symbol
                    min_distance = dist
                    bits = self.symbol_map[best_symbol]
            buffer = bits << decoded_bits | buffer
            decoded_bits += self.symbol_len
            while decoded_bits >= 8:
                self.port_out.put(buffer & 0xff) 
                buffer = buffer >> 8
                decoded_bits -= 8

