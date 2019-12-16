CXX = g++
LIBS = -lm -lpthread
CXXFLAGS = -std=c++11 -fPIC -fopenmp
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -Weffc++ -D_GLIBCXX_DEBUG
else
CXXFLAGS += -O3 -march=native
endif

ALL=dotproduct-omp.exe dotproduct-thread.exe 

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


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
