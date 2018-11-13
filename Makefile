CXX = g++
CXXFLAGS = -std=c++11 -Wall
DEPS = ising.hpp
OBJ1 = ising.o isingSim.o

%.o : %.cpp $(DEPS)
		$(CXX) -c -o $@ $< $(CXXFLAGS)

ising : $(OBJ1)
	    $(CXX) -o $@ $^ -g

clean :
		rm -f *.o $~

.PHONY : all clean
