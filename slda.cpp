
#include <cstdlib>
#include <iostream>

#include "SupervisedLDA.hpp"
#include "ProgressVisitor.hpp"


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

    lda.fit(X, y);

    return 0;
}
