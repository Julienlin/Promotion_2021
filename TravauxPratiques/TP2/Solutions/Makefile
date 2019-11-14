CXX = mpic++
LIBS = -lm 
CXXFLAGS = -std=c++11 -fPIC  -fopenmp
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
else
CXXFLAGS += -O3 -march=native -Wall
endif

SOURCES=$(wildcard *.cpp)
ALL=$(patsubst %.cpp,%.exe,$(SOURCES))

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^


help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"
	@echo "    ALL      :    $(ALL)"


# Rapport
%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css -o $@
