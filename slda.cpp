
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>

#include <docopt/docopt.h>

#include "IEStep.hpp"
#include "IMStep.hpp"
#include "NumpyFormat.hpp"
#include "ProgressVisitor.hpp"
#include "LDABuilder.hpp"
#include "LDA.hpp"


double accuracy_score(const VectorXi &y_true, const VectorXi &y_pred) {
    double accuracy = 0.0;

    for (int i=0; i<y_pred.rows(); i++) {
        if (y_pred(i) == y_true(i)) {
            accuracy += 1.0/y_pred.rows();
        }
    }

    return accuracy;
}


void save_lda(
    std::string model_path,
    typename LDA<double>::LDAState lda_state
) {
    std::fstream model(
        model_path,
        std::ios::out | std::ios::binary
    );

    model << numpy_format::NumpyOutput<int>(
        lda_state.ids,
        std::vector<size_t>{5, 1},
        true
    );

    for (int i=0; i<5; i++) {
        model << numpy_format::NumpyOutput<double>(
            lda_state.parameters[i].data(),
            std::vector<size_t>{lda_state.parameters[i].size(), 1},
            true
        );
    }

    model << numpy_format::NumpyOutput<double>(*lda_state.alpha);
    model << numpy_format::NumpyOutput<double>(*lda_state.beta);
    model << numpy_format::NumpyOutput<double>(*lda_state.eta);
}


LDA<double> load_lda(std::string model_path, size_t iterations=20) {
    // we will be needing those
    LDA<double>::LDAState lda_state;
    numpy_format::NumpyInput<double> ni;
    VectorXd alpha;
    MatrixXd beta;
    MatrixXd eta;

    // open the file
    std::fstream model(
        model_path,
        std::ios::in | std::ios::binary
    );

    // read the implementation ids
    model >> ni;
    std::memcpy(lda_state.ids, ni.data(), ni.shape()[0] * sizeof(int));
    for (int i=0; i<5; i++) {
        model >> ni;
        lda_state.parameters[i].resize(ni.shape()[0]);
        std::memcpy(lda_state.parameters[i].data(), ni.data(), ni.shape()[0] * sizeof(double));
    }

    model >> ni;
    alpha = ni;
    lda_state.alpha = &alpha;

    model >> ni;
    beta = ni;
    lda_state.beta = &beta;

    model >> ni;
    eta = ni;
    lda_state.eta = &eta;

    return LDA<double>(lda_state, beta.rows(), iterations);
}


class TrainingProgress : public IProgressVisitor<double>
{
    public:
        TrainingProgress() {
            em_iterations_ = 1;
            currently_in_expectation_ = false;
        }

        void visit(Progress<double> progress) {
            switch (progress.state) {
                case Expectation:
                    // we are coming from Maximization so output the global
                    // iteration number
                    if (!currently_in_expectation_) {
                        std::cout << std::endl
                                  << "E-M Iteration " << em_iterations_
                                  << std::endl;
                        em_iterations_ ++;
                    }

                    // update the flags and member variables with the progress
                    currently_in_expectation_ = true;
                    likelihood_ = progress.value;

                    // if we have seen 100 iterations print out a progress
                    if ((progress.partial_iteration+1) % 100 == 0) {
                        std::cout << progress.partial_iteration+1 << std::endl;
                    }
                    break;
                case Maximization:
                    // we are coming from Expectation so output the computed
                    // log likelihood
                    if (currently_in_expectation_) {
                        std::cout << "Likelihood: " << likelihood_ << std::endl;
                    }
                    currently_in_expectation_ = false;
                    std::cout << "log p(y | \\bar{z}, eta): " << -progress.value << std::endl;
                    break;
                case IterationFinished:
                    break;
            }
        }

    private:
        int em_iterations_;
        bool currently_in_expectation_;
        double likelihood_;
};


class SnapshotEvery : public IProgressVisitor<double>
{
    public:
        SnapshotEvery(std::string path, int every=10)
            : path_(std::move(path)), every_(every), seen_so_far_(0)
        {}

        void visit(Progress<double> progress) {
            switch (progress.state) {
                case IterationFinished:
                    seen_so_far_ ++;
                    if (seen_so_far_ % every_ == 0) {
                        snapsot(progress.lda_state);
                    }
                    break;
                default:
                    break;
            }
        }

        void snapsot(typename LDA<double>::LDAState lda_state) {
            std::stringstream actual_path;
            actual_path << path_ << "_";
            actual_path.fill('0');
            actual_path.width(3);
            actual_path << seen_so_far_;

            save_lda(actual_path.str(), lda_state);
        }

    private:
        std::string path_;
        int every_;
        int seen_so_far_;
};


class BroadcastVisitor : public IProgressVisitor<double>
{
    public:
        BroadcastVisitor(
            std::vector<std::shared_ptr<IProgressVisitor<double> > > visitors
        ) : visitors_(std::move(visitors))
        {}

        void visit(Progress<double> progress) {
            for (auto visitor : visitors_) {
                visitor->visit(progress);
            }
        }

    private:
        std::vector<std::shared_ptr<IProgressVisitor<double> > > visitors_;
};

void parse_input_data(std::string data_path, MatrixXi &X, MatrixXi &y) {
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

static const char * USAGE =
R"(Supervised LDA and other flavors of LDA.

    Usage:
        slda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                   [--m_step_iterations=MI] [--e_step_tolerance=ET]
                   [--m_step_tolerance=MT] [--fixed_point_iterations=FI]
                   [--regularization_penalty=L] [-q | --quiet]
                   [--snapshot_every=N] DATA MODEL
        slda transform MODEL DATA OUTPUT
        slda evaluate MODEL DATA
        slda (-h | --help)

    Options:
        -h, --help         Show this help
        -q, --quiet         Produce no output to the terminal
        --topics=K          How many topics to train [default: 100]
        --iterations=I      Run LDA for I iterations [default: 20]
        --e_step_iterations=EI  The maximum number of iterations to perform
                                in the E step [default: 10]
        --e_step_tolerance=ET   The minimum accepted relative increase in log
                                likelihood during the E step [default: 1e-4]
        --m_step_iterations=MI  The maximum number of iterations to perform
                                in the M step [default: 20]
        --m_step_tolerance=MT   The minimum accepted relative increase in log
                                likelihood during the M step [default: 1e-4]
        --fixed_point_iterations=FI  The number of fixed point iterations to compute
                                     \phi [default: 20]
        -L L, --regularization_penalty=L  The regularization penalty for the Multinomiali
                                          Logistic Regression [default: 0.05]
        --snapshot_every=N      Snapshot the model every N iterations [default: -1]
)";

int main(int argc, char **argv) {

    std::map<std::string, docopt::value> args = docopt::docopt(
        USAGE,
        {argv+1, argv + argc},
        true,  // show help if requested
        "Supervised LDA 0.1"
    );

    if (args["train"].asBool()) {
        // create the lda model
        LDA<double> lda =
            LDABuilder<double>().
                set_topics(args["--topics"].asLong()).
                set_iterations(args["--iterations"].asLong()).
                set_e_step(
                    IEStep<double>::BatchSupervised,
                    args["--e_step_iterations"].asLong(),
                    std::stof(args["--e_step_tolerance"].asString()),
                    args["--fixed_point_iterations"].asLong()
                ).
                set_m_step(
                    IMStep<double>::BatchSupervised,
                    args["--m_step_iterations"].asLong(),
                    std::stof(args["--m_step_tolerance"].asString()),
                    std::stof(args["--regularization_penalty"].asString())
                );
        
        MatrixXi X, y;
        // Parse data from input file
        parse_input_data(args["DATA"].asString(), X, y);

        std::vector<std::shared_ptr<IProgressVisitor<double> > > visitors;
        if (!args["--quiet"].asBool()) {
            visitors.push_back(std::make_shared<TrainingProgress>());
        }

        if (args["--snapshot_every"].asLong() > 0) {
            visitors.push_back(std::make_shared<SnapshotEvery>(
                args["MODEL"].asString(),
                args["--snapshot_every"].asLong()
            ));
        }

        lda.set_progress_visitor(
            std::make_shared<BroadcastVisitor>(visitors)
        );

        lda.fit(X, y);

        save_lda(args["MODEL"].asString(), lda.get_state());
    } 
    else if (args["transform"].asBool()) {
        MatrixXi X, y;
        // Parse data from input file
        parse_input_data(args["DATA"].asString(), X, y);

        // Load LDA model from file
        LDA<double> lda = load_lda(args["MODEL"].asString());

        MatrixXd doc_topic_distribution = lda.transform(X);
        numpy_format::save(args["OUTPUT"].asString(), doc_topic_distribution);
    } 
    else if (args["evaluate"].asBool()){
        MatrixXi X, y;
        // Parse data from input file
        parse_input_data(args["DATA"].asString(), X, y);

        // Load LDA model from file
        LDA<double> lda = load_lda(args["MODEL"].asString());
        MatrixXi y_predicted = lda.predict(X);
        std::cout << "Accuracy score: " << accuracy_score(y, y_predicted) << std::endl;
    }
    else {
        std::cout << "Not implemented yet" << std::endl;
    }

    return 0;
}
