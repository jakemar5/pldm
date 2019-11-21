#include <iostream>
#include <complex>
#include <mpi.h>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>

// Input Params
// TODO: Put input params into .in file and read in?

int ntraj = 2;                  // Number of total trajectories
int init_state = 0;             // Initial states of forward and backward propogators.
int init_statet = 0;
const int nsteps = 1000;              // Number of SLOW steps to run trajectory
const double nlittle = 100;              // Number of FAST steps
const int nbath = 2;                  // Number of baths
const int nosc = 1000;                // Number of oscillators in harmonic bath
const int nstate = 2;           // Number of system states
const int ncopies = 2;          // How many copies of the trajectories to run (1 forward and 1 backward, for example)
double epsilon = 100.;          // Spin energy gap in wavenumbers
double delta = 100.;            // Coupling between spin states
double lambda = 25.;            // Reorganization energy
double cutoff_freq = 105.;      // Cutoff frequency for spectral density
double temperature = 77.;       // Temp. in Kelvin
double runtime = 1000.;         // Total time to run trajectory over in fs
double hamiltonian[2][2] = {{100., 50.}, {50., 0.0}};  // Where the system is coupled to the bath.


// CONSTANTS
const double PI = 3.14159265358979323;
const double BOLTZMANN = 1.38064852e-23;
const double LIGHT = 3.0e10;      // units cm/s
std::complex<double> eye = sqrt( -1.0 );

double beta = 1./temperature;   // Conversion to atomic energy accounts for Boltzmann constant.
double dt = runtime/nsteps;

// CONVERSIONS
double wavenumber2AtomicAngFreq = 4.556335253e-6;
double kelvin2atomicEnergy = 3.166811506e-6;
double fs2AtomicTime = 41.341374575751;

// FUNCTION PROTOTYPES
void convert_parameter_units();
void get_mapping_init_positions_and_momenta(double q[ncopies][nstate], double p[ncopies][nstate], int &num_procs);
void get_bath_information(double omega[], double c[]);
void get_bath_initial_conditions(double q_bath[nbath][nosc], double p_bath[nbath][nosc], double omega[], double &beta);
void propagate(double q_map[ncopies][nstate], double p_map[ncopies][nstate], double &dt, int system_copy);
double uniform_random_number(double = 0., double = 1.);
double gaussian_random_number(double mean, double sigma);



int main(int argc, char *argv[]) {
  convert_parameter_units();
  std::ofstream outfile;
  outfile.open("populations.out");
  outfile << "#Time(s)\tPopulation\n";

  std :: cout << "Hamiltonian:\n";
  std :: cout << hamiltonian[0][0] << '\t' << hamiltonian[0][1] << '\n';
  std :: cout << hamiltonian[1][0] << '\t' << hamiltonian[1][1] << '\n';

  MPI_Init(0, 0); // Initialize MPI Processes
  int num_procs, my_pe;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
  srand(my_pe + time(0)); // Set random seed


  double omega[nosc];
  double c[nosc];
  double positions_bath[nbath][nosc];
  double momenta_bath[nbath][nosc];

  get_bath_information(omega, c);

  double positions_map[ncopies][nstate];
  double momenta_map[ncopies][nstate];

  double dt_little = dt/nlittle;
  double force;

  // Each processor to run n-trajectories
  for (int itraj=0; itraj<ntraj/num_procs; itraj++) {
      // Get initial conditions for each trajectory

      get_mapping_init_positions_and_momenta(positions_map, momenta_map, num_procs);
      get_bath_initial_conditions(positions_bath, momenta_bath, omega, beta);

      for (int istep = 0; istep < nsteps; istep++) {        // TODO: Do I need this loop over baths?
          // Loop over nuclear steps
          force = 0.5 * (0.5 * c[0] *
                         (pow(positions_map[0][0], 2) + pow(momenta_map[0][0], 2) - pow(positions_map[0][1], 2) -
                          pow(momenta_map[0][1], 2)
                          + pow(positions_map[1][0], 2) + pow(momenta_map[1][0], 2) - pow(positions_map[1][1], 2) -
                          pow(momenta_map[1][1], 2))
                         + (2 * pow(omega[0], 2) * positions_bath[0][0]));

          for (int ibath = 0; ibath < nbath; ibath++) {
              for (int iosc = 0; iosc < nosc; iosc++) {
                  // Loop over electron steps

                  propagate(positions_map, momenta_map, dt_little, 0);
                  propagate(positions_map, momenta_map, dt_little, 1);

                  // Advance each oscillator

                  momenta_bath[ibath][iosc] += 0.5 * force * dt;
                  positions_bath[ibath][iosc] += momenta_bath[ibath][iosc] * dt;

                  force = 0.5 * (0.5 * c[iosc] * (pow(positions_map[0][0], 2) + pow(momenta_map[0][0], 2) -
                                                  pow(positions_map[0][1], 2) - pow(momenta_map[0][1], 2)
                                                  + pow(positions_map[1][0], 2) + pow(momenta_map[1][0], 2) -
                                                  pow(positions_map[1][1], 2) - pow(momenta_map[1][1], 2))
                                 + (2 * pow(omega[iosc], 2) * positions_bath[ibath][iosc]));

                  momenta_bath[ibath][iosc] += 0.5 * force * dt;

                  // Write populations to file?     pop_k = |k><k| = a_k^\dagger * a_k = 0.5 * (q_k^2 + p_k^2 - 1)
                  // Do we already have this as momenta_map and positions_map? Do this step inside of 'propagate' function?
                  // Average forward and backward propagators?
              }
          }
      }
  }

  MPI_Finalize();
  outfile.close();
  return 0;
}


void get_mapping_init_positions_and_momenta(double q[ncopies][nstate], double p[ncopies][nstate], int &num_procs) {
  // Sample init. q's and p's to start trajectories.

  for (int icopy=0; icopy<ncopies; icopy++) {

      for (int istate = 0; istate < nstate; istate++) {
          q[icopy][istate] = gaussian_random_number(0, 1);
          p[icopy][istate] = gaussian_random_number(0, 1);
          //std::cout << "q(" << icopy << "," << istate << ") = " << q[icopy][istate] << "\tp(" << icopy << "," << istate << ") = " << p[icopy][istate] << '\n';
      }
  }
  std::complex<double> initial_weight = 0.5*(q[0][init_state] - eye * p[0][init_state])*(q[1][init_statet] + eye * p[1][init_statet]); // TODO: Where does this weight get applied?
  return;
}

void get_bath_information(double omega[], double c[]) {
    // frequencies - omega       coupling strengths - c.

    double w_max = 1500. * wavenumber2AtomicAngFreq;
    double w_min = 0. * wavenumber2AtomicAngFreq;
    double dw = (w_max-w_min)/nosc;
    //std::cout << "dw = " << dw << '\n';


    for (int iosc=0; iosc<nosc; iosc++) {
        omega[iosc] = w_min + dw * (iosc+1);
        c[iosc] = pow((4. * lambda / PI) * (pow(omega[iosc],2.) / cutoff_freq)/(1. + pow((omega[iosc]/ cutoff_freq),2.)) * dw, 0.5);
        //std::cout << "omega" << iosc << " = " << omega[iosc] << "\t\tc" << iosc << " = " << c[iosc] << '\n';
    }
    return;
}

void get_bath_initial_conditions(double q_bath[nbath][nosc], double p_bath[nbath][nosc], double omega[nosc], double &beta) {
    // Sample Wigner Transformed Boltzmann Operator for Simple H.O.
    // Loop over all baths and all oscillators in each bath.
    for (int ibath=0; ibath<nbath; ibath++) {
        for (int iosc=0; iosc<nosc; iosc++){
            p_bath[ibath][iosc] = gaussian_random_number(0, sqrt(omega[iosc] / (2.0 * tanh(0.5*beta*omega[iosc]))));
            q_bath[ibath][iosc] = gaussian_random_number(0, sqrt(omega[iosc] / (2.0 * tanh(0.5*beta*omega[iosc]))) / omega[iosc]);
        }
    }
}

void propagate(double q_map[ncopies][nstate], double p_map[ncopies][nstate], double &dt, int system_copy) {
    // Propagate electronic DOFs over short time step.

    double dxdt[nstate], dpdt[nstate];

    for (int istep=0; istep<nlittle; istep++) {
        // Calculate time derivatives
        for (int istate=0; istate < nstate; istate++) {             // TODO: Should these loops be over # of system states or # of copies?
            dxdt[istate] = 0.;
            dpdt[istate] = 0.;

            for (int jstate=0; jstate < nstate; jstate++) {
                dxdt[istate] += p_map[system_copy][jstate]*hamiltonian[istate][jstate];         // TODO: Should there be a factor of 0.5 in off-diagonal term?
                dpdt[istate] += -q_map[system_copy][jstate]*hamiltonian[istate][jstate];
            }
        }
        // Advance electronic DOFs
        for (int istate=0; istate<nstate; istate++) {
            q_map[system_copy][istate] += dxdt[istate] * dt;
            p_map[system_copy][istate] += dpdt[istate] * dt;
        }
    }
}



double uniform_random_number(double min, double max) {
  // Generate pseudo random number between a minimum and maximum value.

  double unif_rand_num = min + (max-min)*rand()/RAND_MAX;
  return unif_rand_num;
}

double gaussian_random_number(double mu, double sigma) {
  // Generate Gaussian random number with mean=mu, std_dev=sigma

  double gaus_rand_num = std::sqrt(-2.*log(uniform_random_number())) * cos(2*PI*uniform_random_number());
  return sigma*gaus_rand_num + mu;
}

void convert_parameter_units() {

    temperature *= kelvin2atomicEnergy;         // Kelvin -> Atomic Energy
    runtime *= fs2AtomicTime;                   // Total time to run trajectory over
    epsilon *= wavenumber2AtomicAngFreq;        // Spin energy gap in wavenumbers
    delta *= wavenumber2AtomicAngFreq;          // Coupling between spin states
    lambda *= wavenumber2AtomicAngFreq;         // Reorganization energy
    cutoff_freq *= wavenumber2AtomicAngFreq;    // Cutoff frequency for spectral density
    dt *= fs2AtomicTime;                        // Time step for propagation.
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            hamiltonian[i][j] *= wavenumber2AtomicAngFreq;
        }
    }

}