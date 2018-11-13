#include "ising.hpp"

using namespace std;

int main() {
    LoschmidtSim L(1., 1., 1e-5, 5, 1.);
    cout << "Initialized\n";
    vector<complex<double> > myRes = L.run(1000, 2);
    cout << "Ran simulations\n";
    ofstream ampFile;
    ampFile.open("/media/kai/TOSHIBA EXT/StochasticQuantum/results/testRun.txt");
    for (int i = 0; i < int(myRes.size()); i++) {
        ampFile << 1e-5*100*i << " " << myRes[i] << "\n";
    }
    ampFile.close();
    cout << "Wrote data file\n";
    return 0;
}
