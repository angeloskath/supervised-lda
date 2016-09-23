#include "applications/utils.hpp"

namespace utils {


Eigen::VectorXd create_class_weights(const Eigen::VectorXi & y) {
    int C = y.maxCoeff() + 1;
    Eigen::VectorXd Cy = Eigen::VectorXd::Zero(C);

    for (int d=0; d<y.rows(); d++) {
        Cy[y[d]]++;
    }

    Cy = y.rows() / (Cy.array() * C).array();

    return Cy;
}

}  // namespace utils
