#include "../include/floquet.hpp"
#include "../include/stabilizer.hpp"
#include "../include/pauli_product.hpp"
#include "../libs/pcg-cpp/include/pcg_random.hpp"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <math.h>
#include <random>

using namespace std;

int main(int argc, char *argv[])
{

    int dx = 7;
    int dy = 2;
    int const n_t = 50;
    int const n_simu = 100;
    int nq = dx * dy * 2;
    unsigned int lx;
    unsigned int lgz = 0;
    double theta1_uni;
    double theta2;
    double shift;
    double error_rate;
    string filename;
    double measur1, measur2;
    double measur1mat[n_simu][n_t] = {0};
    double measur2mat[n_simu][n_t] = {0};
    double measur1list[n_t] = {0};
    double measur2list[n_t] = {0};

    pcg_extras::seed_seq_from<random_device> seed_source;

    pcg32 engine1(seed_source);
    pcg32 engine2(seed_source);

    uniform_real_distribution<double> dist1(-0.01, 0.01);
    uniform_real_distribution<double> dist2(-0.1, 0.1);

    error_rate = 0.02;
    shift = 0.01;

    ofstream outfile;
    filename = string("data/nodecoder_multi_pcg_") + string("dx=") + to_string(dx) + string("_ns=") + to_string(n_simu) + "_nt=" + to_string(n_t) + string(".dat");

    outfile.open(filename);

    for (int i = 1; i < 2 * dy; i = i + 2)
    {
        lgz = lgz | (1 << (i * dx + 1));
    }

#pragma omp parallel for num_threads(20) schedule(dynamic)
    for (int simu = 0; simu < n_simu; simu++)
    {
        cx_dvec psi = initial_allzero(dx, dy);

        for (int t = 1; t < n_t; t++)
        {

            apply_stabl_uniform(dx, dy, psi);

            // add errors
            for (int i = 0; i < nq; i++)
            {
                lx = 1 << i;
                // random_value = dist1(engine1) / 100.0;
                theta2 = error_rate * M_PI + dist1(engine1);
                apply_ppr(lx, 0, theta2, psi);
            }

            if (t % 2 == 0)
            {
                measur1 = -measure_pp(0, lgz, psi);
            }
            else
            {
                measur1 = measure_pp(0, lgz, psi);
            }

            measur1mat[simu][t] = measur1 / n_simu;

            for (int i = 0; i < dx; i++)
            {
                lx = 1 << (1 * dx + i);
                // random_value = (dist2(engine2) % 200 - 100) / 1000.0;
                theta1_uni = 0.5 * M_PI + shift * M_PI + dist2(engine2);
                apply_ppr(lx, 0, theta1_uni, psi);
            }

            if (t % 2 == 0)
            {
                measur2 = -measure_pp(0, lgz, psi);
            }
            else
            {
                measur2 = measure_pp(0, lgz, psi);
            }

            measur2mat[simu][t] = measur2 / n_simu;
        }
    }

    for (int t = 1; t < n_t; t++)
    {
        for (int simu = 0; simu < n_simu; simu++)
        {
            measur1list[t] += measur1mat[simu][t];
            measur2list[t] += measur2mat[simu][t];
        }
    }

    for (int t = 1; t < n_t; t++)
    {
        outfile << t << " " << measur1list[t] << " " << measur2list[t] << endl;
    }

    outfile.close();
    return 0;
}
