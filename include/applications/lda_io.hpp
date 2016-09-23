#ifndef _APPLICATIONS_LDA_IO_HPP_
#define _APPLICATIONS_LDA_IO_HPP_

#include <memory>

#include <Eigen/Core>

#include "ldaplusplus/Parameters.hpp"

using namespace ldaplusplus;

namespace io {


/**
  * Parse the input data from a file and save them to two Eigen matrixes, one
  * for the input data and one for their corresponding class labels.
  *
  * @param data_path The file to read the input data from
  * @param X An Eigen matix containing the input data
  * @param y An Eigen matrix containing the class labels of the input data
  */
void parse_input_data(
    std::string data_path,
    Eigen::MatrixXi &X,
    Eigen::MatrixXi &y
);

/**
  * Parse the input data from a file and save them to one Eigen matrix that
  * corresponds to the input data.
  *
  * @param data_path The file to read the input data from
  * @param X An Eigen matix containing the input data
  */
void parse_input_data(
    std::string data_path,
    Eigen::MatrixXi &X
);

/**
  * Save a set of model parameters in a file defined by the model_path input
  * argument, according to the NumpyFormat.
  *
  * @param model_path The file to save the set of the input parameters
  * @param parameters The set of input parameters to be saved
  */
void save_lda(
    std::string model_path,
    std::shared_ptr<parameters::Parameters> parameters
);

/**
  * Read a set of model parameters saved in NumpyInput from a file
  *
  * @param model_path The file to read a set of model parameters from
  * @return The model parameters
  */
std::shared_ptr<parameters::SupervisedModelParameters<double> > load_lda(
    std::string model_path
);


}  // namespace io

#endif  // _APPLICATIONS_LDA_IO_HPP_
