#!/bin/python3

import struct
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

x = open(sys.argv[1], 'rb')
size = os.path.getsize(sys.argv[1])
if os.path.splitext(sys.argv[1])[1] in ['.s16']:
    element_size = 2
    num_elements = int(size/element_size)
    f = struct.unpack('h'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(f, '.-')
elif os.path.splitext(sys.argv[1])[1] in ['.fc32', '.cf32']:
    element_size = 4
    num_elements = int(size/element_size)
    f = struct.unpack('f'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(f[::2], '.-')
    plt.plot(f[1::2], '.-')
elif os.path.splitext(sys.argv[1])[1] in ['.f32']:
    element_size = 4
    num_elements = int(size/element_size)
    f = struct.unpack('f'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(f, '.-')
elif os.path.splitext(sys.argv[1])[1] in ['.fc64']:
    element_size = 8
    num_elements = int(size/element_size)
    f = struct.unpack('d'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(f[::2], '.-')
    plt.plot(f[1::2], '.-')
elif os.path.splitext(sys.argv[1])[1] in ['.fft']:
    element_size = 8
    num_elements = int(size/element_size)
    f = struct.unpack('d'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(np.abs(f[::2]), '.-')
    plt.plot(np.angle(f[1::2]), '.-')
elif os.path.splitext(sys.argv[1])[1] in ['.f3c32', '.c3f32']:
    element_size = 4
    num_elements = int(size/element_size)
    f = struct.unpack('f'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(f[::3], '.-')
    plt.plot(f[1::3], '.-')
    plt.plot(f[2::3], '.-')
elif os.path.splitext(sys.argv[1])[1] in ['.const']:
    element_size = 4
    num_elements = int(size/element_size)
    f = struct.unpack('f'*num_elements, x.read(size))
    plt.figure(0)
    cmap = range(len(f[1::2]))
    cmap = list(map(lambda x: (x / len(f[1::2])) / (x / len(f[1::2]) + 1), cmap))
    plt.scatter(f[::2], f[1::2], c=cmap, cmap='magma_r')
elif os.path.splitext(sys.argv[1])[1] in ['.sym']:
    element_size = 4
    num_elements = int(size/element_size)
    f = struct.unpack('f'*num_elements, x.read(size))
    plt.figure(0)
    plt.plot(np.abs(f[::2]), '.-')
    plt.plot(np.angle(f[1::2]), '.-')
    sps = 80
    num_taps = 1011
    num_symbols = (len(f)//2)//sps
    for i in range(num_symbols):
        plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, f[i*sps+num_taps//2]], color='red')
plt.title(sys.argv[1])
plt.grid(True)
plt.show()

