#include <chrono>
#include <iostream>

#include <Eigen/Core>

#include "e_step_utils.hpp"

using namespace Eigen;


int main(int argc, char **argv) {
    VectorXd X = VectorXd::Random(1000);
    X.array() -= X.minCoeff();
    X.array() /= X.sum();
    VectorXi X_counts = (X*1000).cast<int>();

    MatrixXd beta = MatrixXd::Random(600, 1000);
    beta.array() -= beta.minCoeff();
    beta.array().rowwise() /= beta.array().colwise().sum();

    MatrixXd eta = MatrixXd::Random(600, 10);

    VectorXd gamma = VectorXd::Random(600);
    gamma.array() -= gamma.minCoeff();
    gamma.array() /= gamma.sum();
    gamma *= 1000;

    MatrixXd phi_old(600, 1000);
    MatrixXd phi = MatrixXd::Random(600, 1000);
    phi.array() -= phi.minCoeff();
    phi.array().rowwise() /= phi.array().colwise().sum();

    // Warm up the cache
    for (int i=0; i<10; i++) {
        e_step_utils::compute_supervised_approximate_phi<double>(
            X,
            0,
            beta,
            eta,
            gamma,
            phi
        );
    }

    std::chrono::high_resolution_clock clock;
    std::chrono::high_resolution_clock::duration s(0);
    for (int i=0; i<100; i++) {
        auto start = clock.now();
        e_step_utils::compute_supervised_approximate_phi<double>(
            X,
            0,
            beta,
            eta,
            gamma,
            phi
        );
        s += clock.now() - start;
    }

    std::cout << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > >(s).count() << "s" << std::endl;

    return 0;
}
