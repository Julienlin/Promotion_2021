CXX = g++
LIBS = -lm -lpthread
CXXFLAGS = -std=c++11 -fPIC -fopenmp
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -Weffc++ -D_GLIBCXX_DEBUG
else
CXXFLAGS += -O3 -march=native
endif

ALL=dotproduct-omp.exe dotproduct-thread.exe \
    TestProduct.exe TestProduct_opt.exe TestProduct_blockomp.exe \
    bhudda-omp.exe
# bitonic.exe bitonic_thread.exe

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# dependances particulieres :
TestProduct.exe: Matrix.cpp ProdMatMat.cpp
TestProduct_opt.exe: Matrix.cpp ProdMatMat_opt.cpp
TestProduct_blockomp.exe: Matrix.cpp ProdMatMat_blockomp.cpp
bitonic.exe: Vecteur.cpp
bitonic_thread.exe: Vecteur.cpp


help:
	@echo "Available targets:"
	@echo "    all: compile all executables"
	@echo "       $(ALL)"
	@echo "Configuration:"
	@echo "    CXX     : $(CXX)"
	@echo "    CXXFLAGS: $(CXXFLAGS)"
	@echo "Add DEBUG=yes to compile in debug"


# Rapport
%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css -o $@
