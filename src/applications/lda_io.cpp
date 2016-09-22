#include <iostream>
#include <fstream>

#include "ldaplusplus/NumpyFormat.hpp"

#include "applications/lda_io.hpp"

using namespace ldaplusplus;

namespace io {


void parse_input_data(
    std::string data_path,
    Eigen::MatrixXi &X,
    Eigen::MatrixXi &y
) {
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

void parse_input_data(
    std::string data_path,
    Eigen::MatrixXi &X
) {
    // read the data in
    std::fstream data(
        data_path,
        std::ios::in | std::ios::binary
    );
    numpy_format::NumpyInput<int> ni;

    // read the data
    data >> ni;
    X = ni;
}

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

std::shared_ptr<parameters::SupervisedModelParameters<double> > load_lda(
    std::string model_path
) {
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

