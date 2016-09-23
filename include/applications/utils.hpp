#ifndef _APPLICATIONS_UTILS_HPP_
#define _APPLICATIONS_UTILS_HPP_

#include <Eigen/Core>

namespace utils {


Eigen::VectorXd create_class_weights(const Eigen::VectorXi & y);

}  // namespace utils

#endif // _APPLICATIONS_UTILS_HPP_
