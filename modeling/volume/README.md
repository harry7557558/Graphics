`.raw` file: unsigned little endian, size = `NX*NY*NZ*bytes_per_voxel`

`.info` file:

    NX NY NZ
    bytes_per_voxel map_to_0 map_to_1
    R11 R12 R13 Tx
    R21 R22 R23 Ty
    R31 R32 R33 Tz

Normal must face outward, achieved by either a greater `map_to_0` or a negative matrix determinant.

See [triangulate.cpp](triangulate.cpp) for details.
