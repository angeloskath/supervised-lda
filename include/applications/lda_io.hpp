#ifndef LDA_IO_H
#define LDA_IO_H

#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include "ldaplusplus/NumpyFormat.hpp"
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
void parse_input_data(std::string data_path, Eigen::MatrixXi &X, Eigen::MatrixXi &y) {
    // read the data in
    std::fstream data(
        data_path,
        std::ios::in | std::ios::binary
    );
    numpy_format::NumpyInput<int> ni;

    // read the data
    data >> ni;
    X = ni;

    // and the labels
    data >> ni;
    y = ni;
}

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
) {
    // cast the model parameters SupervisedModelParameters regardless the type
    // of the trained LDA model. In this way, one can train initially a
    // unsupervised LDA and then continue the training in a supervised manner
    auto model_parameters =
        std::static_pointer_cast<parameters::SupervisedModelParameters<double> >(
            parameters
        );

    // we 'll save the model here
    std::fstream model(
        model_path,
        std::ios::out | std::ios::binary
    );

    model << numpy_format::NumpyOutput<double>(model_parameters->alpha);
    model << numpy_format::NumpyOutput<double>(model_parameters->beta);
    model << numpy_format::NumpyOutput<double>(model_parameters->eta);
}

/**
  * Read a set of model parameters saved in NumpyInput from a file
  *
  * @param model_path The file to read a set of model parameters from
  * @return The model parameters
  */
std::shared_ptr<parameters::SupervisedModelParameters<double> > load_lda(std::string model_path) {
    // we will be needing those
    auto model_parameters = std::make_shared<parameters::SupervisedModelParameters<double> >();
    numpy_format::NumpyInput<double> ni;

    // open the file
    std::fstream model(
        model_path,
        std::ios::in | std::ios::binary
    );

    model >> ni; model_parameters->alpha = ni;
    model >> ni; model_parameters->beta = ni;
    model >> ni; model_parameters->eta = ni;

    return model_parameters;
}


}  // namespace io

#endif // LDA_IO_H
