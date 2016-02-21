
#include "SupervisedLDA.hpp"

int main(int argc, char **argv) {
    SupervisedLDA<float> lda(100);

    MatrixXi X = MatrixXi::Random(100, 10);
    VectorXi y = VectorXi::Random(100);

    lda.partial_fit(X, y);

    return 0;
}
