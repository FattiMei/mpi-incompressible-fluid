#!/bin/bash
cd ../build
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc ..
make fftw_Dirichlet
rm ../MaxErrDirich.txt
## rm ../MaxErrNeumann.txt
## make fftw_Neumann
for ((i = 3; i < 100; i++)); do
    ./fftw_Dirichlet $i >> ../MaxErrDirich.txt
   ##  ./fftw_Neumann   $i >> ../MaxErrNeumann.txt
   tail ../MaxErrDirich.txt
    echo " Sono al $i esimo di 99 "
done



