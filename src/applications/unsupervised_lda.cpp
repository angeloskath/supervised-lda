#include <iostream>

#include <Eigen/Core>

#include "applications/lda_io.hpp"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Incorrect number of arguments." << std::endl;
        std::cout << "Usage: " << *argv << "[input_file] [output_file]"
                  << std::endl;
        return 1;
    }

    Eigen::MatrixXi X, y;
    // Parse data from input file
    io::parse_input_data(argv[1], X, y);
}

