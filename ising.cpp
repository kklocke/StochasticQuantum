#include "ising.hpp"


/*
 * SpinChain default initialization
 * We default to a chain of 5 spins with J and h both set to unity
 * Default time step is consistent with the choice from the Doyon paper.
 */
SpinChain::SpinChain() {
    T = 0.;
    N = 5;
    J = 1.;
    h = 1.;
    dt = 1e-5;
    Xi = vector<double>(3*N, 0);
}

/*
 * SpinChain custom initialization
 * Can specify the choice of coupling strength J, transverse
 * field h, time discretization dt, and chain length N.
 */
SpinChain::SpinChain(double myJ, double myH, double myDT, int NumSpins) {
    assert(NumSpins > 0);
    assert(myDT > 0);
    T = 0.;
    N = NumSpins;
    J = myJ;
    h = myH;
    dt = myDT;
    Xi = vector<double>(3*N, 0);
}

SpinChain::~SpinChain() {}


/*
 * SpinChain reset
 * Sets the time back to zero and resets the Xi vector to all zeros.
 */
void SpinChain::reset() {
    T = 0.;
    Xi = vector<double>(3*N, 0.);
}


/*
 * SpinChain update
 * Compute the matrixes D and M, then apply a single update state accordingly
 */
void SpinChain::update() {
    // fill in later
    T += dt;
    // should keep the values of J(k) precomputed when i initialize the spin chain
    vector<double> D(3*N, 0.);
    for (int i = 0; i < N; i++) {
        D[3*i] = h * 0.5 * (1 - pow(Xi[3*j],2));
        D[3*i + 1] = -h * Xi[3*i];
        D[3*i + 2] = h * 0.5 * exp(Xi[3*j + 1]);
    }
    // Define the values of the matrix Chi here
    // Use that to construct matrix M from the notes
    // Update all of the Xi values
    // We can precompute much of M, just need to do the random sampling each time
}

/*
 * SpinChain getXi
 * Returns the current Xi. Can construct the time evolution
 * operator from the values contained in Xi.
 */
vector<double> SpinChain::getXi() {
    return Xi;
}

/*
 * LoschmidtSim default initialization
 * Calls the default SpinChain initializer
 * Uses the Doyon time discretization
 */
LoschmidtSim::LoschmidtSim() {
    T = 1.;
    dt = 1e-5;
    myChain = new SpinChain();
}

/*
 * LoschmidtSim custom initialization
 * Can set the parameters for the SpinChain.
 */
LoschmidtSim::LoschmidtSim(double myJ, double myH, double myDT, int myN, double myT) {
    assert(myN > 0);
    assert(myDT > 0);
    assert(myT > 0);
    T = myT;
    dt = myDT;
    myChain = new SpinChain(myJ, myH, myDT, myN);
}

/*
 * LoschmidtSim destructor
 * Frees the dynamically allocated SpinChain
 */
LoschmidtSim::~LoschmidtSim() {
    free myChain;
}

/*
 * LoschmidtSim reset function
 * Just calls the reset function for the SpinChain members.
 */
void LoschmidtSim::reset() {
    myChain->reset();
}

/*
 * LoschmidtSim run function
 * Runs the simulation up to time T
 * Returns the Loschmidt amplitude at each time step
 */
vector<double> LoschmidtSim::run(int saveIter) {
    assert(saveIter > 0);
    int numSteps = int(T / dt);
    vector<double> ret(int(numSteps / saveIter), 0.);
    vector<double> myXi = myChain->getXi();
    for (int i = 0; i < numSteps; i++) {
        myChain->update();
        if (i % saveIter == 0) {
            myXi = myChain->getXi();
            double myAmp = 0.;
            for (int j = 1; j < int(myXi.size()); j+=3) {
                myAmp += myXi[j];
            }
            myAmp = exp(-myAmp / 2.);
            ret[int(i / saveIter)] = myAmp;
        }
    }
    myChain->reset();
    return ret;
}


vector<double> LoschmidtSim::run(int saveIter, int numSims) {
    assert(numSims > 0);
    vector<double> ret = run(saveIter);
    for (int i = 1; i < numSims; i++) {
        vector<double> tmp = run(saveIter);
        for (int j = 0; j < int(ret.size()); j++) {
            ret[j] += tmp[j];
        }
    }
    for (int i = 0; i < int(ret.size()); i++) {
        ret[i] /= float(numSims);
    }
    return ret;
}
