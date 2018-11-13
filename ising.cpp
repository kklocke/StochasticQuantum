#include "ising.hpp"

default_random_engine generator;
normal_distribution<double> distribution(0.,1.0);

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
    Xi = vector<complex<double> >(3*N, complex<double>(0.,0.));
    int indMin = int(float(N)/2.)+1;
    JVals = vector<vector<complex<double> > >(N, vector<complex<double> >(N-indMin+1, complex<double>(0.,0.)));
    for (int j = 1; j <= N; j++) {
        for (int m = indMin; m <= N; m++) {
            double kj = float(j)*M_PI*((2*float(m) / float(N)) - 1.);
            JVals[j-1][m-indMin] = sqrt(8*J*cos(kj/float(j))/N)*complex<double>(cos(kj), -sin(kj));
        }
    }
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
    Xi = vector<complex<double> >(3*N, complex<double>(0.,0.));
    int indMin = int(float(N)/2.)+1;
    JVals = vector<vector<complex<double> > >(N, vector<complex<double> >(N-indMin+1, complex<double>(0.,0.)));
    for (int j = 1; j <= N; j++) {
        for (int m = indMin; m <= N; m++) {
            double kj = float(j)*M_PI*((2*float(m) / float(N)) - 1.);
            JVals[j-1][m-indMin] = sqrt(8.*J*cos(kj/float(j))/N)*complex<double>(cos(kj), -sin(kj));
        }
    }
}

SpinChain::~SpinChain() {}


/*
 * SpinChain reset
 * Sets the time back to zero and resets the Xi vector to all zeros.
 */
void SpinChain::reset() {
    T = 0.;
    for (int i = 0; i < int(Xi.size()); i++) {
        Xi[i] = complex<double>(0.,0.);
    }
}


/*
 * SpinChain update
 * Compute the matrixes D and M, then apply a single update state accordingly
 */
void SpinChain::update() {
    // fill in later
    T += dt;
    // should keep the values of J(k) precomputed when i initialize the spin chain
    vector<complex<double> > D(3*N, complex<double>(0.,0.));
    vector<complex<double> > Chi(3*N, complex<double>(0.,0.));
    for (int i = 0; i < N; i++) {
        D[3*i] = h * 0.5 * (1. - pow(Xi[3*i],2));
        D[3*i + 1] = -h * Xi[3*i];
        D[3*i + 2] = h * 0.5 * exp(Xi[3*i + 1]);
        Chi[3*i] = Xi[3*i];
        Chi[3*i + 1] += 1.;
    }
    for (int i = 0; i < 3*N; i++) {
        complex<double> dXi(0.,0.);
        dXi += D[i];
        for (int k = 0; k < int(JVals[0].size()); k++) {
            dXi += Chi[i]*JVals[int(i/3)][k]*distribution(generator);
        }
        dXi /= complex<double>(0.,1.);
        dXi *= dt;
        Xi[i] += dXi;
    }
}

/*
 * SpinChain getXi
 * Returns the current Xi. Can construct the time evolution
 * operator from the values contained in Xi.
 */
vector<complex<double> > SpinChain::getXi() {
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
    // delete(myChain);
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
 * Returns the Loschmidt amplitude every saveIter steps.
 */
vector<complex<double> > LoschmidtSim::run(int saveIter) {
    assert(saveIter > 0);
    int numSteps = int(T / dt);
    vector<complex<double> > ret(int(numSteps / saveIter)+1, complex<double>(0.,0.));
    cout << "Set up the simulation\n";
    for (int i = 0; i < numSteps; i++) {
        myChain->update();
        if (i % saveIter == 0) {
            ret[int(i / saveIter)] = getAmp();
        }
        if (i % 5000 == 0) {
            cout << dt * i << "\t" << getAmp() << endl;
        }
    }
    cout << "Finished the update loop\n";
    myChain->reset();
    cout << "Reset the chain\n";
    cout << ret[0];
    return ret;
}

complex<double> LoschmidtSim::getAmp() {
    vector<complex<double> > myXi = myChain->getXi();
    complex<double> myAmp(0.,0.);
    for (int j = 1; j < int(myXi.size()); j+=3) {
        myAmp += myXi[j];
    }
    myAmp = exp(-myAmp / 2.);
    return myAmp;
}

/*
 * LoschmidtSim run function
 * Runs the simulation up to time T
 * Saves the Loschmidt amplitude every saveIter steps.
 * Averages over numSim instances of the simulation.
 */
vector<complex<double> > LoschmidtSim::run(int saveIter, int numSims) {
    assert(numSims > 0);
    cout << "Sim iter: 1\n";
    vector<complex<double> > ret = run(saveIter);
    for (int i = 1; i < numSims; i++) {
        cout << "Sim iter: " << i+1 << "\n";
        vector<complex<double> > tmp = run(saveIter);
        for (int j = 0; j < int(ret.size()); j++) {
            ret[j] += tmp[j];
        }
    }
    for (int i = 0; i < int(ret.size()); i++) {
        ret[i] /= float(numSims);
    }
    return ret;
}
