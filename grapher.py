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
elif os.path.splitext(sys.argv[1])[1] in ['.fc32', '.cf32', '.f32']:
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
plt.title(sys.argv[1])
plt.grid(True)
plt.show()

