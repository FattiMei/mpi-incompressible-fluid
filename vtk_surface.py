import sys
import struct
import numpy as np
import matplotlib.pyplot as plt


def get_face_stride(face):
    origin = face[0]
    direction = face[1] - face[0]

    i = 1
    while i < len(face):
        d = face[i] - face[i-1]

        if np.allclose(d, direction):
            i += 1
        else:
            break

    return i


def remove_duplicate_points(face):
    _, idx = np.unique(face[:,0:3], axis=0, return_index=True)

    return face[idx]


def extract_common_coordinate(arr):
    count = {}

    for i in arr:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1

    return list(count.keys())[np.argmax(list(count.values()))]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python vtk_surface.py <vtk input file>")
        sys.exit(1)

    FILENAME = sys.argv[1]

    with open(FILENAME, 'rb') as file:
        data = file.read()

    point_header_start = data.find(b'POINTS')
    point_header_end   = point_header_start + data[point_header_start:].find(b'\n')
    point_header       = data[point_header_start:point_header_end]

    _, npoints, dtype = point_header.split()
    npoints = int(npoints)

    if dtype == b'float':
        sizeof = 4
        typechar = 'f'

    elif dtype == b'double':
        sizeof = 8
        typechar = 'd'

    else:
        ValueError("Invalid point type")


    point_data_start = point_header_end + 1
    point_data_end   = point_data_start + 3*sizeof*npoints

    points = np.array(
        struct.unpack('>' + 3*npoints*typechar, data[point_data_start:point_data_end])
    )
    points = points.reshape((npoints, 3))
    points = np.column_stack((points, np.arange(npoints)))

    closest_x = extract_common_coordinate(points[:,0])
    closest_y = extract_common_coordinate(points[:,1])
    closest_z = extract_common_coordinate(points[:,2])

    yz_face = remove_duplicate_points(np.array(sorted(points[points[:,0] == closest_x].tolist())))
    xz_face = remove_duplicate_points(np.array(sorted(points[points[:,1] == closest_y].tolist())))
    xy_face = remove_duplicate_points(np.array(sorted(points[points[:,2] == closest_z].tolist())))

    plt.scatter(xz_face[:,1], xz_face[:,2])
    plt.show()

    plt.scatter(xz_face[:,0], xz_face[:,1])
    plt.show()

    plt.scatter(xz_face[:,0], xz_face[:,2])
    plt.show()

    yz_stride = get_face_stride(yz_face[:,0:3])
    xz_stride = get_face_stride(xz_face[:,0:3])
    xy_stride = get_face_stride(xy_face[:,0:3])

    yz_rows = len(yz_face) // yz_stride
    xz_rows = len(xz_face) // xz_stride
    xy_rows = len(xy_face) // xy_stride

    yz_face = yz_face.reshape((yz_rows, yz_stride, 4))
    xz_face = xz_face.reshape((xz_rows, xz_stride, 4))

    cells = []
    for i in range(yz_face.shape[0]-1):
        for j in range(yz_face.shape[1]-1):
            cells.append((
                4,
                yz_face[i, j, 3],
                yz_face[i, j+1, 3],
                yz_face[i+1, j+1, 3],
                yz_face[i+1, j, 3]
            ))

    for i in range(xz_face.shape[0]-1):
        for j in range(xz_face.shape[1]-1):
            cells.append((
                4,
                xz_face[i, j, 3],
                xz_face[i, j+1, 3],
                xz_face[i+1, j+1, 3],
                xz_face[i+1, j, 3]
            ))

    try:
        xy_face = xy_face.reshape((xy_rows, xy_stride, 4))
        for i in range(xy_face.shape[0]-1):
            for j in range(xy_face.shape[1]-1):
                cells.append((
                    4,
                    xy_face[i, j, 3],
                    xy_face[i, j+1, 3],
                    xy_face[i+1, j+1, 3],
                    xy_face[i+1, j, 3]
                ))
    except:
        print("Can't assemble XY face")

    byte_cells = bytearray().join(
        struct.pack(
            '>iiiii',
           int(c[0]),
           int(c[1]),
           int(c[2]),
           int(c[3]),
           int(c[4])
        )
        for c in cells
    )

    cell_types = bytearray().join(
        struct.pack('>i', 9)
        for _ in range(len(cells))
    )

    point_data_header_start = data.find(b'POINT_DATA')

    cell_info = bytearray().join([
        bytes(f'CELLS {len(cells)} {5 * len(cells)}\n', 'ascii'),
        byte_cells,
        bytes(f'\nCELL_TYPES {len(cells)}\n', 'ascii'),
        cell_types,
        b'\n'
    ])

    with open('surface.vtk', 'wb') as file:
        file.write(data[:point_data_header_start])
        file.write(cell_info)
        file.write(data[point_data_header_start:])
