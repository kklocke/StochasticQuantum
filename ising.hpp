#include <vector>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
// #include <stdio>
#include <complex>
#include <assert.h>

using namespace std;
// using namespace std::literals;

#define _USE_MATH_DEFINES

/* SpinChain class
 * Base class for running simulations of 1-D transverse field Ising model
 * Stores the disentangling variables in the vector Xi
 * Can set the interaction and external field strengths, as well as
 * the size of the spin chain and the time discretization.
 */
class SpinChain
{
public:
    SpinChain();
    SpinChain(double myJ, double myH, double myDT, int myN);
    ~SpinChain();
    void reset();
    void update();
    vector<complex<double> > getXi();
private:
    double T, J, h, dt;
    vector<vector<complex<double> > > JVals;
    int N;
    vector<complex<double> > Xi;
};

/*
 * LoschmidtSim class
 * Class for running simulations to extract the
 * Loschmidt Amplitude from a SpinChain member.
 */
class LoschmidtSim
{
public:
    LoschmidtSim();
    LoschmidtSim(double myJ, double myH, double myDT, int myN, double myT);
    ~LoschmidtSim();
    void reset();
    vector<complex<double> > run(int saveIter);
    vector<complex<double> > run(int saveIter, int numSims);
    complex<double> getAmp();
private:
    double T, dt;
    SpinChain *myChain;
};
