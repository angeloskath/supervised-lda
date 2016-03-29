
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <utility>

#include <docopt/docopt.h>

#include "Events.hpp"
#include "IEStep.hpp"
#include "IMStep.hpp"
#include "LDABuilder.hpp"
#include "LDA.hpp"
#include "NumpyFormat.hpp"
#include "Parameters.hpp"
#include "ProgressEvents.hpp"


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
    std::shared_ptr<Parameters> parameters
) {
    // cast the model parameters to the model parameters type this script deals
    // with, namely SupervisedModelParameters
    auto model_parameters =
        std::static_pointer_cast<SupervisedModelParameters<double> >(
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


std::shared_ptr<SupervisedModelParameters<double> > load_lda(std::string model_path) {
    // we will be needing those
    auto model_parameters = std::make_shared<SupervisedModelParameters<double> >();
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


class TrainingProgress : public IEventListener
{
    public:
        TrainingProgress() {
            e_iterations_ = 0;
            m_iterations_ = 0;
            em_iterations_ = 1;
            output_em_step_ = true;
            likelihood_ = 0;
            cnt_likelihoods_ = 0;
        }

        void on_event(std::shared_ptr<Event> event) {
            if (event->id() == "ExpectationProgressEvent") {
                auto progress = std::static_pointer_cast<ExpectationProgressEvent<double> >(event);

                // output the EM count
                if (e_iterations_ == 0) {
                    std::cout << "E-M Iteration " << em_iterations_ << std::endl;
                }

                // if we have seen 100 iterations print out a progress
                e_iterations_++;
                if (e_iterations_ % 100 == 0) {
                    std::cout << e_iterations_ << std::endl;
                }

                // keep track of the likelihood
                if (std::isfinite(progress->likelihood()) && progress->likelihood() < 0) {
                    likelihood_ += progress->likelihood();
                    cnt_likelihoods_++;
                }
            }
            else if (event->id() == "MaximizationProgressEvent") {
                auto progress = std::static_pointer_cast<MaximizationProgressEvent<double> >(event);

                std::cout << "log p(y | \\bar{z}, eta): " << progress->likelihood() << std::endl;

                m_iterations_++;
            }
            else if (event->id() == "EpochProgressEvent") {
                if (likelihood_ < 0) {
                    std::cout << "Per document likelihood: " <<
                        likelihood_ / cnt_likelihoods_ << std::endl << std::endl;
                }

                // reset the member variables
                likelihood_ = 0;
                cnt_likelihoods_ = 0;
                em_iterations_ ++;
                e_iterations_ = 0;
                m_iterations_ = 0;
            }
        }

    private:
        int e_iterations_;
        int m_iterations_;
        int em_iterations_;
        double likelihood_;
        int cnt_likelihoods_;
        bool output_em_step_;
};


class SnapshotEvery : public IEventListener
{
    public:
        SnapshotEvery(std::string path, int every=10)
            : path_(std::move(path)), every_(every), seen_so_far_(0)
        {}

        void on_event(std::shared_ptr<Event> event) {
            if (event->id() == "EpochProgressEvent") {
                auto progress = std::static_pointer_cast<EpochProgressEvent<double> >(event);

                seen_so_far_ ++;
                if (seen_so_far_ % every_ == 0) {
                    snapsot(progress->model_parameters());
                }
            }
        }

        void snapsot(std::shared_ptr<Parameters> parameters) {
            std::stringstream actual_path;
            actual_path << path_ << "_";
            actual_path.fill('0');
            actual_path.width(3);
            actual_path << seen_so_far_;

            save_lda(actual_path.str(), parameters);
        }

    private:
        std::string path_;
        int every_;
        int seen_so_far_;
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

VectorXd create_class_weights(const VectorXi & y) {
    int C = y.maxCoeff() + 1;
    VectorXd Cy = VectorXd::Zero(C);

    for (int d=0; d<y.rows(); d++) {
        Cy[y[d]]++;
    }

    Cy = y.rows() / (Cy.array() * C).array();

    return Cy;
}

LDA<double> create_lda_for_training(
    std::map<std::string, docopt::value> &args,  // should be const but const
                                                 // C++ map is annoying
    const MatrixXi & X,
    const VectorXi & y
) {
    LDABuilder<double> builder;

    // trivial parameters
    builder.set_iterations(args["--iterations"].asLong());
    builder.set_workers(args["--workers"].asLong());

    // Choose the e step
    if (args["--semi_supervised"].asBool()) {
        builder.set_e(builder.get_semi_supervised_e_step(
            builder.get_supervised_e_step(
                args["--e_step_iterations"].asLong(),
                std::stof(args["--e_step_tolerance"].asString()),
                args["--fixed_point_iterations"].asLong()
            ),
            builder.get_fast_classic_e_step(
                args["--e_step_iterations"].asLong(),
                std::stof(args["--e_step_tolerance"].asString())
            )
        ));
    } else if (args["--unsupervised_e_step"].asBool()) {
        builder.set_e(builder.get_fast_classic_e_step(
            args["--e_step_iterations"].asLong(),
            std::stof(args["--e_step_tolerance"].asString())
        ));
    } else if (args["--fast_e_step"].asBool()) {
        builder.set_e(builder.get_fast_supervised_e_step(
            args["--e_step_iterations"].asLong(),
            std::stof(args["--e_step_tolerance"].asString())
        ));
    } else if (args["--multinomial"].asBool()) {
        builder.set_e(builder.get_supervised_multinomial_e_step(
            args["--e_step_iterations"].asLong(),
            std::stof(args["--e_step_tolerance"].asString()),
            std::stof(args["--mu"].asString())
        ));
    } else {
        builder.set_e(builder.get_supervised_e_step(
            args["--e_step_iterations"].asLong(),
            std::stof(args["--e_step_tolerance"].asString()),
            args["--fixed_point_iterations"].asLong()
        ));
    }

    // Choose the m step
    if (args["--semi_supervised"].asBool()) {
        builder.set_semi_supervised_batch_m_step(
            args["--m_step_iterations"].asLong(),
            std::stof(args["--m_step_tolerance"].asString()),
            std::stof(args["--regularization_penalty"].asString())
        );
    } else if (args["--online_m_step"].asBool()) {
        builder.set_supervised_online_m_step(
            create_class_weights(y),
            std::stof(args["--regularization_penalty"].asString()),
            args["--batch_size"].asLong(),
            std::stof(args["--momentum"].asString()),
            std::stof(args["--learning_rate"].asString()),
            std::stof(args["--beta_weight"].asString())
        );
    } else if (args["--multinomial"].asBool()) {
        builder.set_supervised_multinomial_m_step(
            std::stof(args["--mu"].asString())
        );
    } else {
        builder.set_supervised_batch_m_step(
            args["--m_step_iterations"].asLong(),
            std::stof(args["--m_step_tolerance"].asString()),
            std::stof(args["--regularization_penalty"].asString())
        );
    }

    // Initialize the model parameters
    if (args["--continue"]) {
        auto model = load_lda(args["--continue"].asString());
        builder.
            initialize_topics_from_model(model).
            initialize_eta_from_model(model);
    } else {
        builder.
            initialize_topics("seeded", X, args["--topics"].asLong()).
            initialize_eta("zeros", X, y, args["--topics"].asLong());
    }

    return builder;
}

static const char * USAGE =
R"(Supervised LDA and other flavors of LDA.

    Usage:
        slda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                   [--m_step_iterations=MI] [--e_step_tolerance=ET]
                   [--m_step_tolerance=MT] [--fixed_point_iterations=FI]
                   [--multinomial] [--mu=MU]
                   [--unsupervised_e_step] [--fast_e_step]
                   [--online_m_step] [--semi_supervised]
                   [--regularization_penalty=L] [--beta_weight=BW]
                   [--momentum=MM] [--learning_rate=LR] [--batch_size=BS]
                   [-q | --quiet] [--snapshot_every=N] [--workers=W]
                   [--continue=M] DATA MODEL
        slda transform [-q | --quiet] [--e_step_iterations=EI]
                       [--e_step_tolerance=ET] [--workers=W]
                       MODEL DATA OUTPUT
        slda evaluate [-q | --quiet] [--e_step_iterations=EI]
                      [--e_step_tolerance=ET] [--workers=W]
                      MODEL DATA
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
        --unsupervised_e_step   Use the unsupervised E step to calculate phi and gamma
        --fast_e_step           Choose a variant of E step that doesn't compute
                                likelihood in order to be faster
        --online_m_step         Choose online M step that updates the model
                                parameters after seeing mini_batch documents
        --semi_supervised       Train a semi supervised lda
        --m_step_iterations=MI  The maximum number of iterations to perform
                                in the M step [default: 200]
        --m_step_tolerance=MT   The minimum accepted relative increase in log
                                likelihood during the M step [default: 1e-4]
        --fixed_point_iterations=FI  The number of fixed point iterations to compute
                                     \phi [default: 20]
        --multinomial           Use the multinomial version of supervised LDA
        --mu=MU                 The multinomial prior on the naive bayesian
                                classification [default: 2]
        -L L, --regularization_penalty=L  The regularization penalty for the Multinomial
                                          Logistic Regression [default: 0.05]
        --beta_weight=BW        Set the weight of the previous beta parameters
                                w.r.t to the new from the minibatch [default: 0.9]
        --momentum=MM           Set the momentum for changing eta [default: 0.9]
        --learning_rate=LR      Set the learning rate for changing eta [default: 0.01]
        --batch_size=BS         The mini-batch size for the online learning [default: 128]
        --snapshot_every=N      Snapshot the model every N iterations [default: -1]
        --workers=N             The number of concurrent workers [default: 1]
        --continue=M            A model to continue training from
)";

int main(int argc, char **argv) {

    std::map<std::string, docopt::value> args = docopt::docopt(
        USAGE,
        {argv+1, argv + argc},
        true,  // show help if requested
        "Supervised LDA 0.1"
    );

    if (args["train"].asBool()) {
        MatrixXi X, y;
        // Parse data from input file
        parse_input_data(args["DATA"].asString(), X, y);

        auto lda = create_lda_for_training(args, X, y);

        if (!args["--quiet"].asBool()) {
            lda.get_event_dispatcher()->add_listener<TrainingProgress>();
        }

        if (args["--snapshot_every"].asLong() > 0) {
            lda.get_event_dispatcher()->add_listener<SnapshotEvery>(
                args["MODEL"].asString(),
                args["--snapshot_every"].asLong()
            );
        }

        lda.fit(X, y);

        save_lda(args["MODEL"].asString(), lda.model_parameters());
    } 
    else if (args["transform"].asBool()) {
        MatrixXi X, y;
        // Parse data from input file
        parse_input_data(args["DATA"].asString(), X, y);

        // Load LDA model from file
        auto model = load_lda(args["MODEL"].asString());
        LDABuilder<double> b;
        LDA<double> lda = b.set_workers(args["--workers"].asLong()).
            set_e(b.get_fast_classic_e_step(
                args["--e_step_iterations"].asLong(),
                std::stof(args["--e_step_tolerance"].asString())
            )).
            initialize_topics_from_model(model).
            initialize_eta_from_model(model);

        if (!args["--quiet"].asBool()) {
            lda.get_event_dispatcher()->add_listener<TrainingProgress>();
        }

        MatrixXd doc_topic_distribution = lda.transform(X);
        numpy_format::save(args["OUTPUT"].asString(), doc_topic_distribution);
    } 
    else if (args["evaluate"].asBool()){
        MatrixXi X, y;
        // Parse data from input file
        parse_input_data(args["DATA"].asString(), X, y);

        // Load LDA model from file
        auto model = load_lda(args["MODEL"].asString());
        LDABuilder<double> b;
        LDA<double> lda = b.set_workers(args["--workers"].asLong()).
            set_e(b.get_fast_classic_e_step(
                args["--e_step_iterations"].asLong(),
                std::stof(args["--e_step_tolerance"].asString())
            )).
            initialize_topics_from_model(model).
            initialize_eta_from_model(model);

        if (!args["--quiet"].asBool()) {
            lda.get_event_dispatcher()->add_listener<TrainingProgress>();
        }

        MatrixXi y_predicted = lda.predict(X);
        std::cout << "Accuracy score: " << accuracy_score(y, y_predicted) << std::endl;
    }
    else {
        std::cout << "Not implemented yet" << std::endl;
    }

    return 0;
}
