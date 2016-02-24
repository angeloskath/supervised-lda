
#include <cstdlib>
#include <iostream>

#include "SupervisedLDA.hpp"
#include "ProgressVisitor.hpp"

void split_training_test_set(
    const MatrixXi &X,
    const VectorXi &y,
    MatrixXi &train_X,
    MatrixXi &test_X,
    VectorXi &train_Y,
    VectorXi &test_Y) 
{
    auto n = X.cols()/2;
    train_X.array() = X.block(0, 0, X.rows(), n);
    test_X.array() = X.block(0, n, X.rows(), X.cols()-n);
    train_Y.array() = y.segment(0, n);
    train_Y.array() = y.segment(n, n);

}

double accuracy_score(const VectorXi &y_true, const VectorXi &y_pred) {
    double accuracy = 0.0;

    for (int i=0; i<y_pred.rows(); i++) {
        if (y_pred(i) == y_true(i)) {
            accuracy += 1.0/y_pred.rows();
        }
    }

    return accuracy;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Usage:\n\t" << argv[0] << " TOPICS ITER" << std::endl;
        return 1;
    }

    SupervisedLDA<double> lda(std::atoi(argv[1]), std::atoi(argv[2]));

    int D, V;
    std::cin >> D >> V;

    MatrixXi X(V, D);
    VectorXi y(D);

    for (int d=0; d<D; d++) {
        std::cin >> y(d);
        for (int v=0; v<V; v++) {
            std::cin >> X(v, d);
        }
    }
    MatrixXi X_train(V, D/2);
    VectorXi y_train(D/2);
    MatrixXi X_test(V, D/2);
    VectorXi y_test(D/2);
    
    split_training_test_set(X, y, X_train, X_test, y_train, y_test);

    double likelihood;
    bool expectation;
    lda.set_progress_visitor(std::make_shared<FunctionVisitor<double> >([&](Progress<double> p){
        switch (p.state) {
            case Expectation:
                expectation = true;
                likelihood = p.value;
                if ((p.partial_iteration+1) % 100 == 0) {
                    std::cout << p.partial_iteration+1 << std::endl;
                }
                break;
            case Maximization:
                if (expectation) {
                    std::cout << "Likelihood: " << likelihood << std::endl;
                }
                expectation = false;
                std::cout << "log p(y | \\bar{z}, eta): " << -p.value << std::endl;
                break;
        }
    }));

    lda.fit(X_train, y_train);
    VectorXi train_predictions = lda.predict(X_train);
    VectorXi test_predictions = lda.predict(X_test);

    std::cout << "Training Accuracy: " << accuracy_score(y_train, train_predictions) << std::endl;
    std::cout << "Test Accuracy: " << accuracy_score(y_test, test_predictions) << std::endl;

    return 0;
}
