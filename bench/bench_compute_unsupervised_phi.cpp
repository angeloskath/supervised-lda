#include <chrono>
#include <iostream>

#include <Eigen/Core>

#include "ldaplusplus/e_step_utils.hpp"

using namespace Eigen;
using namespace ldaplusplus;


int main(int argc, char **argv) {
    VectorXd gamma = VectorXd::Random(600);
    gamma.array() -= gamma.minCoeff();
    gamma.array() /= gamma.sum();
    gamma.array() *= 1000;

    MatrixXd beta = MatrixXd::Random(600, 1000);
    beta.array() -= beta.minCoeff();
    beta.array().rowwise() /= beta.array().colwise().sum();

    MatrixXd phi(beta.rows(), beta.cols());

    // Warm up the cache
    for (int i=0; i<10; i++) {
        e_step_utils::compute_unsupervised_phi<double>(beta, gamma, phi);
    }

    std::chrono::high_resolution_clock clock;
    std::chrono::high_resolution_clock::duration s(0);
    for (int i=0; i<100; i++) {
        auto start = clock.now();
        e_step_utils::compute_unsupervised_phi<double>(beta, gamma, phi);
        s += clock.now() - start;
    }

    std::cout << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > >(s).count() << "s" << std::endl;

    return 0;
}
