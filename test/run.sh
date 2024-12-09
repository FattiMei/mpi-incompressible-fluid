#!/bin/bash
for ((i = 3; i < 10; i++)); do
    cd ../test
    sed -i -e "s|////// INSERT VALUE OF N|N = $val" fftw3D_test.cpp
    cd ../build
    cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc ..
    make fftw_test
    ./fftw_test3D > ../MaxErrDirich.txt
    echo " Sono al $i esimo N"
done



