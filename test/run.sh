#!/bin/bash
cd ../build
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc ..
make fftw_test
for ((i = 3; i < 100; i++)); do
    ./fftw_test3D $i >> ../MaxErrDirich.txt
    echo " Sono al $i esimo N"
done



