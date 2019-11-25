CXX = g++
LIBS = -lm -lpthread
CXXFLAGS = -std=c++11 -fPIC  -fopenmp
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -Weffc++
else
CXXFLAGS += -O3 -march=native
endif

ALL=TestProduct.exe dotproduct.exe bitonic.exe bhudda.exe

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

TestProduct.exe: Matrix.cpp ProdMatMat.cpp
bitonic.exe: Vecteur.cpp
bitonicJD.exe: Vecteur.cpp
bitonicXJ.exe: Vecteur.cpp


help:
	@echo "Available targets : "
	@echo "    all             : compile all executables"
	@echo "    dotproduct.exe  : Compile dot product executable"
	@echo "    TestProduct.exe : Compile matrix-matrix product executable"
	@echo "    bitonic.exe     : Compile bitonic sort example executable"
	@echo "    bhudda.exe      : Compile bhuddabrot set executable"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"


# Rapport
%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css -o $@
