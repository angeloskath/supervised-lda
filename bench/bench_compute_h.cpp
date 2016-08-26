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

    MatrixXd eta = MatrixXd::Random(600, 10);

    MatrixXd phi = MatrixXd::Random(600, 1000);
    phi.array() -= phi.minCoeff();
    phi.array().rowwise() /= phi.array().colwise().sum();

    VectorXd h(phi.rows());

    // Warm up the cache
    for (int i=0; i<10; i++) {
        e_step_utils::compute_h<double>(X_counts, X, eta, phi, h);
    }

    std::chrono::high_resolution_clock clock;
    std::chrono::high_resolution_clock::duration s(0);
    for (int i=0; i<100; i++) {
        auto start = clock.now();
        e_step_utils::compute_h<double>(X_counts, X, eta, phi, h);
        s += clock.now() - start;
    }

    std::cout << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > >(s).count() << "s" << std::endl;

    return 0;
}
