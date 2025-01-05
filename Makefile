# Istruzioni per la giuria: questo è il makefile che presentiamo in alternativa alla build cmake, è preferibile che si usi quest'ultima perchè CMake is so nice :)
#
# Nella prima sezione ci sono le variabili da editare a mano:
#   CXX è il compilatore, deve essere mpixx perché nel programma usiamo le funzionalità di MPI.
#   Preferiremmo usare la versione più aggiornata del compilatore perché usiamo funzionalità di C++23
#
# Questa build non sarà delle più robuste in termini di incremental build, ma va bene per fare una full build del progetto
CXX = mpicxx -std=c++23

CXX_FLAGS = -Wall -Wextra -Ofast -march=native -mtune=native -funroll-all-loops -flto -fno-signed-zeros -fno-trapping-math -flto=auto

DEFINES = -DNDEBUG -DUSE_DOUBLE=1

DECOMP_DIR = ./deps/2Decomp_C
DECOMP_SRC = $(DECOMP_DIR)/Alloc.cpp     \
	     $(DECOMP_DIR)/Best2DGrid.cpp    \
	     $(DECOMP_DIR)/C2Decomp.cpp      \
	     $(DECOMP_DIR)/Halo.cpp          \
	     $(DECOMP_DIR)/IO.cpp            \
	     $(DECOMP_DIR)/MemSplitMerge.cpp \
	     $(DECOMP_DIR)/TransposeX2Y.cpp  \
	     $(DECOMP_DIR)/TransposeY2X.cpp  \
	     $(DECOMP_DIR)/TransposeY2Z.cpp  \
	     $(DECOMP_DIR)/TransposeZ2Y.cpp  \

DECOMP_OBJ = $(patsubst $(DECOMP_DIR)/%.cpp, mbuild/decomp/%.o, $(DECOMP_SRC))

MIF_DIR = ./src
MIF_SRC = $(wildcard $(MIF_DIR)/*.cpp)
MIF_OBJ = $(patsubst $(MIF_DIR)/%.cpp, mbuild/mif/%.o, $(MIF_SRC))

INCLUDE = -I ./include -I $(DECOMP_DIR)

# those variables need to be defined from the outside, else fallback to those
FFTW_INC ?= /usr/include/
FFTW_LIB ?= /usr/lib/x86_64-linux-gnu/

INCLUDE += -I $(FFTW_INC)
LIBS += -lfftw3 -lfftw3f -L $(FFTW_LIB)

# decommentare questa riga nel caso si abbiano problemi di compilazione (in particolare riferimento alle funzionalità di std::byteswap
# DEFINES += -DMIF_LEGACY_COMPILER


all: mbuild mif


mif: $(MIF_OBJ) mbuild/decomp.a
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)


mbuild/mif/%.o: $(MIF_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(DEFINES) $(INCLUDE) -c -o $@ $^


mbuild/decomp.a: $(DECOMP_OBJ)
	ar rcs $@ $^


mbuild/decomp/%.o: $(DECOMP_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(DEFINES) $(INCLUDE) -c -o $@ $^


mbuild:
	mkdir -p mbuild
	mkdir -p mbuild/decomp
	mkdir -p mbuild/mif


.PHONY: clean
clean:
	rm mif mbuild/decomp/* mbuild/mif/*


# sometimes if `solution.vtk` is already present in the root folder, the file will be badly overwritten
# when in doubt of the consistency of the results, just call this rule
resclean:
	rm -f *.vtk *.dat
