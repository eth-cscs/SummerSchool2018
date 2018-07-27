# Build OpenACC version

```bash
module switch PrgEnv-cray PrgEnv-pgi
module load craype-accel-nvidia60
make
```

NOTE: The current version can only be compiled by PGI due to the implicit attach operation required when copying `Field` internally to the GPU.
For a version that would work with OpenACC 2.0 and the Cray compiler, you need to work with the raw pointers from the `Field` internals by calling either `host_data()` inside an OpenACC data region or by calling `device_data()` and using the `devicptr` clause.


# Performance without RDMA

```
................................................
                            ranks
      dim       1       2       4       8      16
.................................................
 128x128     3101    2123    1687    1342    1103
 256x256     3089    2220    1799    1434    1224
 512x512     3060    2411    2007    1639    1379
1024x1024    1843    2255    2098    1722    1479
2048x2048     638     982    1676    1684    1471
4096x4096     175     286     529     866    1157
8192x8192      37      75     146     282     512
.................................................
```


# Performance with RDMA

To enable, simply call `exchange_rdma()` instead of `exchange()` inside the `operations.cpp`.

```bash
export MPICH_RDMA_ENABLED_CUDA=1
```

```
.................................................
                            ranks
      dim       1       2       4       8      16
.................................................
 128x128     3065    2108    1693    1349    1133
 256x256     3097    2193    1792    1480    1167
 512x512     3041    2369    1970    1639    1375
1024x1024    1839    2059    1956    1717    1500
2048x2048     638    1041    1425    1506    1440
4096x4096     175     334     581     886    1129
8192x8192      37      74     143     265     471
.................................................
```
