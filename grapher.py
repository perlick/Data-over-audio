import struct
import sys
import os
import matplotlib.pyplot as plt

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
    if os.path.splitext(sys.argv[1])[1] in ['.fc32', '.cf32']:
        plt.figure(0)
        plt.plot(f[::2], '.-')
        plt.plot(f[1::2], '.-')
    else:
        plt.figure(0)
        plt.plot(f, '.-')
plt.title(sys.argv[1])
plt.grid(True)
plt.show()

