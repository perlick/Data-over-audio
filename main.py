import numpy as np
from typing import Dict
import math
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.signal import butter,filtfilt

QAM_SYMBOLS_ENCODE = {
    0b00: -1 + -1j,
    0b01: -1 + 1j,
    0b10: 1 + -1j,
    0b11: 1 + 1j
    }
QAM_SYMBOLS_DECODE = {}
for key, val in QAM_SYMBOLS_ENCODE.copy().items():
    QAM_SYMBOLS_DECODE[val] = key

def data_to_symbols(data: bytes, symbol_map: Dict, symbol_len):
    mask = 2**symbol_len - 1
    num_symbols = math.ceil((len(data) * 8) / symbol_len)
    symbol_vec = np.zeros(num_symbols, dtype=np.complex_)
    for i in range(num_symbols):
        first_bit_index = i * symbol_len
        first_byte_index = first_bit_index // 8
        last_bit_intex = first_bit_index + symbol_len
        last_byte_index = last_bit_intex // 8 
        value = int.from_bytes(data[first_byte_index:last_byte_index+1], byteorder='little')
        shift = first_bit_index % 8
        symbol_vec[i] = symbol_map[value >> shift & mask]
    return symbol_vec

def iq_distance(symbol_1, symbol_2):
    i_1 = symbol_1.real
    i_2 = symbol_2.real
    q_1 = symbol_1.imag
    q_2 = symbol_2.imag

    return math.sqrt((i_1 - i_2)**2 + (q_1 - q_2)**2)

def symbols_to_data(symbols, symbol_map, symbol_len):
    data = bytearray()
    decoded_bits = 0
    buffer = 0
    for symbol in symbols:
        min_distance = 1000
        best_symbol = None
        for ref_symbol in symbol_map.keys():
            if isinstance(ref_symbol, complex):
                if best_symbol is None:
                    best_symbol = ref_symbol
                    min_distance = iq_distance(ref_symbol, symbol)
                    continue
                dist = iq_distance(ref_symbol, symbol)
                if dist < min_distance:
                    best_symbol = ref_symbol
                    min_distance = dist
        bits = symbol_map[best_symbol]
        buffer = bits << decoded_bits | buffer
        decoded_bits += symbol_len
        while decoded_bits >= 8:
            data.append(buffer & 0xff) 
            buffer = buffer >> 8
            decoded_bits -= 8 
    if decoded_bits != 0:
        data.append(buffer)
    return data

def symbols_to_signal(symbols, carrier_freq_hz, symbol_rate_hz, sample_rate_hz):
    """
    Normal IQ signal creation.
    I multiplied by sin. Q by cos. then added together to form signal
    symbols: list of symbols to transmit
    carrier_freq_hz: frequency of carrier to upconvert to 
    symbol_rate_hz: symbols/sec on tx side
    sample_rate_hz: rate to sample signal at on rx side
    """
    num_symbols = len(symbols)
    num_samples = int((num_symbols / symbol_rate_hz) * sample_rate_hz)
    t = np.arange(num_samples)/sample_rate_hz
    tx_lo_i = np.sin(2*np.pi*carrier_freq_hz*t)
    tx_lo_q = np.cos(2*np.pi*carrier_freq_hz*t)

    samples_per_symbol = num_samples / num_symbols
    tx_I_sym = np.zeros(num_samples)
    tx_Q_sym = np.zeros(num_samples)
    for i in range(num_samples):
        tx_I_sym[i] = symbols[math.floor(i/num_samples*num_symbols)].real
        tx_Q_sym[i] = symbols[math.floor(i/num_samples*num_symbols)].imag

    tx_I_out = tx_I_sym * tx_lo_i
    tx_Q_out = tx_Q_sym * tx_lo_q

    tx_s_out = tx_I_out + tx_Q_out

    # now pretend to sample this signal with a different lo
    # rx_lo_i = np.sin(2*np.pi*carrier_freq*t)
    # rx_lo_q = np.cos(2*np.pi*carrier_freq*t)

    tx_s_out = tx_s_out + ((np.random.rand(num_samples)-0.5)/10)

    return tx_s_out

def signal_to_wav(signal, sample_rate_hz, file_name):
    scaled = np.int16(signal / np.max(np.abs(signal)) * 32767)
    write(file_name, int(sample_rate_hz), scaled)

def wav_to_signal(file_name):
    a = read(file_name)
    sample_rate_hz = a[0]
    data = np.array(a[1],dtype=float)
    data /= np.max(np.abs(data),axis=0)
    return data, sample_rate_hz

def butter_lowpass_filter(data, cutoff, fs, order=2):
    """ butter loowpass filter has perfect response 
    
    cutoff: # desired cutoff frequency of the filter, Hz
    fs: sample rate of data
    order: Polynomial order of the signal. sin can be approximated as quadratic
    """

    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def signal_to_iq_samples(signal, sample_rate_hz, carrier_freq_hz):
    num_samples = len(signal)
    t = np.arange(num_samples)/sample_rate_hz
    t = t+0.11
    rx_lo_i = np.sin(2*np.pi*carrier_freq_hz*t)
    rx_lo_q = np.cos(2*np.pi*carrier_freq_hz*t)

    rx_I_sym = signal * rx_lo_i
    rx_Q_sym = signal * rx_lo_q

    rx_I_sym = butter_lowpass_filter(rx_I_sym, carrier_freq_hz, sample_rate_hz)
    rx_Q_sym = butter_lowpass_filter(rx_Q_sym, carrier_freq_hz, sample_rate_hz)
    
    out = rx_I_sym + rx_Q_sym*1j
    save_signal_slice_multi([signal, rx_lo_i, rx_lo_q, rx_I_sym, rx_Q_sym], 1, 150, 44e3, "rx_iq_circuit.jpg")
    

    return out

def iq_samples_to_symbols_simple(signal, symbol_rate_hz, sample_rate_hz):
    samples_per_symbol = sample_rate_hz / symbol_rate_hz
    num_symbols = int(len(signal) // samples_per_symbol)
    out = np.zeros(num_symbols, dtype=np.complex_)
    for i in range(num_symbols-1):
        out[i] = signal[int(i*samples_per_symbol + 0.5*samples_per_symbol)]
    return out

def iq_samples_to_symbols_meuller(signal, symbol_rate_hz, sample_rate_hz):
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(signal) + 10, dtype=np.complex_)
    out_rail = np.zeros(len(signal) + 10, dtype=np.complex_) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    sps = sample_rate_hz / symbol_rate_hz
    while i_out < len(signal) and i_in+16 < len(signal):
        out[i_out] = signal[i_in + int(mu)] # grab what we think is the "best" sample
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    return out
    
def save_spectrogram(signal, fft_size, Fs, file_name):
    num_rows = len(signal) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        s = np.hamming(fft_size) * signal[i*fft_size:(i+1)*fft_size]
        spectrogram[i,:] += 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(s)))**2 + 1e-12)
    plt.cla()
    plt.imshow(spectrogram, aspect='auto', extent = [Fs/-2, Fs/2, len(signal)/Fs, 0])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [s]")
    # f = np.arange(Fs/-2, Fs/2, Fs/bins)
    # plt.figure(0)
    # plt.plot(f, S,'.-')
    plt.savefig(file_name)

def save_signal_slice(signal, start_t, num_samples, freq, file_name):
    start_i = int(start_t*freq)
    slice = signal[start_i:start_i+num_samples]
    plt.cla()
    plt.figure(0)
    plt.plot(slice, '.-')
    plt.savefig(file_name)

def save_signal_slice_multi(signals, start_t, num_samples, freq, file_name):
    start_i = int(start_t*freq)
    plt.cla()
    plt.figure(0)
    for signal in signals:
        slice = signal[start_i:start_i+num_samples]
        plt.plot(slice, '.-')
    plt.plot
    plt.savefig(file_name)

def plot_constellation(rx_symbols, file_name):
    plt.cla()
    plt.figure(2)
    plt.scatter(rx_symbols.real,rx_symbols.imag)
    plt.savefig(file_name)

if __name__ == "__main__":

    msg = 'deadbeef' # first 32 symbols are for time sync
    msg += '0123456789abcdeffefebaba'*10 # some random hex for fun
    data = bytes.fromhex(msg)
    # data = bytes.fromhex('ffff0000')
    print(list(data))

    data_symbols = data_to_symbols(data, QAM_SYMBOLS_ENCODE, 2)

    # print(data_symbols)

    sample_rate_hz = 44e3
    symbol_rate_hz = 10
    carrier_freq_hz = 1e3

    signal = symbols_to_signal(data_symbols, carrier_freq_hz, symbol_rate_hz, sample_rate_hz)

    print("tx_signal_len:" + str(len(signal)))

    fft_size = 1024
    save_spectrogram(signal, fft_size, sample_rate_hz, "tx_waterfall.jpg")

    save_signal_slice(signal, 0.795, 100, sample_rate_hz, "tx_signal")

    signal_to_wav(signal, sample_rate_hz, "tx.wav")

    rx_signal, rx_sample_rate_hz = wav_to_signal("Record-2024-0131-121827.wav")
    # rx_signal = signal
    # rx_sample_rate_hz = sample_rate_hz
    print("RX_sample_rate: " + str(rx_sample_rate_hz))
    save_spectrogram(rx_signal, fft_size, rx_sample_rate_hz, "rx_waterfall.jpg")
    save_signal_slice(rx_signal, 1, 150, rx_sample_rate_hz, "rx_signal.jpg")

    iq_samples = signal_to_iq_samples(rx_signal, rx_sample_rate_hz, carrier_freq_hz)
    plot_constellation(iq_samples[int(44e3*5):int(44e3*7)], "rx_iq.jpg")

    rx_symbols = iq_samples_to_symbols_simple(iq_samples, symbol_rate_hz, rx_sample_rate_hz)
    plot_constellation(rx_symbols, "rx_constallation.jpg")
    # print(rx_symbols)

    # print(data_symbols, 50, 16 * 1000)

    data_decoded = symbols_to_data(rx_symbols, QAM_SYMBOLS_DECODE, 2)

    print(list(data_decoded))
