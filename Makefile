# Instructions:
# This is a second build option in case CMake does non work on the cluster. If it works, please use CMake, as it 
# was more thoroughly tested.
# Depending on what is available on the cluster, you may have to change some parts of this file.
# CXX is the compiler used. It is mpicxx because we use MPI features.
# If available, use a modern compiler, as C++23 features are used. For example, g++11.4 does not compile natively.
# If the compiler is too old, you may need to uncomment a flag, as stated in another comment in this file.
# In the case of g++11.4, this fix allows it to compile.
# If you want to use single precision floats instead of doubles (which are the default), change
# -DUSE_DOUBLE=1 to -DUSE_DOUBLE=0.
# The environmenta variables FFTW_INC and FFTW_LIB should be defined. If not, a default path will be searched, but may 
# result in errors if the libraries are not available.
# Finally, the clean and resclean rules are defined. The first removes the executable and compiling folder. The latter
# removes the vtk and dat files. It is suggested to class `make resclean` before each run, to avoid potential errors
# when overwriting files. 

CXX = mpicxx -std=c++23

CXX_FLAGS = -Ofast -march=native -mtune=native -funroll-all-loops -flto -fno-signed-zeros -fno-trapping-math -flto=auto
WARNINGS = -Wall -Wextra 

DEFINES = -DNDEBUG -DUSE_DOUBLE=1

DECOMP_DIR = ./deps/2Decomp_C
DECOMP_SRC = $(DECOMP_DIR)/Alloc.cpp         \
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

# These environmental variables should be defined.
# Otherwise, using fallback paths.
FFTW_INC ?= /usr/include/
FFTW_LIB ?= /usr/lib/x86_64-linux-gnu/

INCLUDE += -I $(FFTW_INC)
LIBS += -lfftw3 -lfftw3f -L $(FFTW_LIB)

# Uncomment this line if the compiler is old and cannot compile due to the std::byteswap function.
# DEFINES += -DMIF_LEGACY_COMPILER


all: mbuild mif


mif: $(MIF_OBJ) mbuild/decomp.a
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)


mbuild/mif/%.o: $(MIF_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(WARNINGS) $(DEFINES) $(INCLUDE) -c -o $@ $^


mbuild/decomp.a: $(DECOMP_OBJ)
	ar rcs $@ $^


mbuild/decomp/%.o: $(DECOMP_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(DEFINES) $(INCLUDE) -c -o $@ $^


mbuild:
	mkdir -p mbuild
	mkdir -p mbuild/decomp
	mkdir -p mbuild/mif


# Clean the build artifacts.
.PHONY: clean
clean:
	rm mif mbuild/decomp/* mbuild/mif/*


# Remove vtk and dat files.
resclean:
	rm -f *.vtk *.dat
