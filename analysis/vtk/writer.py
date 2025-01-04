import numpy as np
import struct


header = [
    b'# vtk DataFile Version 4.0',
    b'foo cube',
    b'BINARY',
    b'DATASET STRUCTURED_POINTS',
    b'DIMENSIONS 3 3 1',
    b'ORIGIN 0 0 0',
    b'SPACING 1 1 1',
    b'POINT_DATA 9',
    b'SCALARS volume_scalars double 1',
    b'LOOKUP_TABLE default'
]

data = bytearray().join(
    struct.pack('>d', x)
    for x in np.arange(9)
)


with open('doublepacked.vtk', 'bw') as file:
    for line in header:
        file.write(line)
        file.write(b'\n')

    file.write(data)
