#include <iostream>

#include <Eigen/Core>
#include <docopt/docopt.h>

#include "ldaplusplus/LDABuilder.hpp"
#include "ldaplusplus/LDA.hpp"
#include "ldaplusplus/NumpyFormat.hpp"

#include "applications/EpochProgress.hpp"
#include "applications/ExpectationProgress.hpp"
#include "applications/lda_io.hpp"
#include "applications/MaximizationProgress.hpp"
#include "applications/SnapshotEvery.hpp"

using namespace ldaplusplus;


LDA<double> create_lda_for_train(
    std::map<std::string, docopt::value> &args,  // should be const but const
                                                 // C++ map is annoying
    const Eigen::MatrixXi & X,
    const Eigen::VectorXi & y
) {
    LDABuilder<double> builder;

    // Start building the LDA model by adding the number of iterations and
    // workers
    builder.set_iterations(args["--iterations"].asLong());
    builder.set_workers(args["--workers"].asLong());

    // Add the parameters regarding the Expectation step
    builder.set_supervised_e_step(
        args["--e_step_iterations"].asLong(),
        std::stof(args["--e_step_tolerance"].asString()),
        args["--fixed_point_iterations"].asLong(),
        std::stof(args["--compute_likelihood"].asString()),
        args["--random_state"].asLong()
    );

    // Add the parameters regarding the Maximization step
    builder.set_supervised_m_step(
        args["--m_step_iterations"].asLong(),
        std::stof(args["--m_step_tolerance"].asString()),
        std::stof(args["--regularization_penalty"].asString())
    );

    // Initialize the model parameters
    if (args["--continue"]) {
        auto model = io::load_lda(args["--continue"].asString());
        builder.
            initialize_topics_from_model(model).
            initialize_eta_from_model(model);

    } else if (args["--continue_from_unsupervised"]) {
        auto model = io::load_lda(args["--continue"].asString());
        builder.
            initialize_topics_from_model(model).
            initialize_eta_zeros(y.maxCoeff() + 1);

    } else {
        builder.
            initialize_topics_seeded(
                X,
                args["--topics"].asLong(),
                30,
                args["--random_state"].asLong()
            ).
            initialize_eta_zeros(y.maxCoeff() + 1);

    }

    // LDABuilder can be implicitly cashed in LDA
    return builder;
}

LDA<double> create_lda_for_transform(
    std::map<std::string, docopt::value> &args,
    std::shared_ptr<parameters::SupervisedModelParameters<double>> model
) {
    LDABuilder<double> builder;

    builder.set_workers(args["--workers"].asLong());

    builder.set_classic_e_step(
        args["--e_step_iterations"].asLong(),
        std::stof(args["--e_step_tolerance"].asString()),
        0.0
    );

    builder.
        initialize_topics_from_model(model).
        initialize_eta_from_model(model);

    // LDABuilder can be implicitly cashed in LDA
    return builder;
}

static const char * USAGE =
R"(Console application for supervised LDA.
    
    Usage:
        slda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                   [--e_step_tolerance=ET] [--fixed_point_iterations=FI]
                   [--random_state=RS] [--compute_likelihood=CL]
                   [--m_step_iterations=MI] [--m_step_tolerance=MT] 
                   [--regularization_penalty=L]
                   [-q | --quiet] [--snapshot_every=N] [--workers=W]
                   [--continue=M] [--continue_from_unsupervised=M] DATA MODEL
        slda transform [-q | --quiet] [--e_step_iterations=EI]
                       [--e_step_tolerance=ET] [--workers=W]
                       MODEL DATA OUTPUT
        slda (-h | --help)

    General Options:
        -h, --help                        Show this help
        -q, --quiet                       Produce no output to the terminal
        --topics=K                        How many topics to train [default: 100]
        --iterations=I                    Run LDA for I iterations [default: 20]
        --random_state=RS                 The initial seed value for any random numbers
                                          needed [default: 0]
        --snapshot_every=N                Snapshot the model every N iterations [default: -1]
        --workers=N                       The number of concurrent workers [default: 1]
        --continue=M                      A model to continue training from
        --continue_from_unsupervised=M    An unsupervised model to continue training from

    E Step Options:
        --e_step_iterations=EI            The maximum number of iterations to perform
                                          in the E step [default: 10]
        --e_step_tolerance=ET             The minimum accepted relative increase in log
                                          likelihood during the E step [default: 1e-4]
        --fixed_point_iterations=FI       The number of fixed point iterations to compute
                                          phi [default: 20]
        --compute_likelihood=CL           The percentage of documents to compute the
                                          likelihood for (1.0 means compute for every
                                          document) [default: 0.0]

    M Step Options:
        --m_step_iterations=MI            The maximum number of iterations to perform
                                          in the M step [default: 200]
        --m_step_tolerance=MT             The minimum accepted relative increase in log
                                          likelihood during the M step [default: 1e-4]
        -L L, --regularization_penalty=L  The regularization penalty for the Multinomial
                                          Logistic Regression [default: 0.05]
)";

int main(int argc, char **argv) {
    
    std::map<std::string, docopt::value> args = docopt::docopt(
        USAGE,
        {argv+1, argv + argc},
        true,  // show help if requested
        "Supervised LDA 0.1"
    );

    if (args["train"].asBool()) {
        
        Eigen::MatrixXi X, y;
        // Parse data from input file
        io::parse_input_data(args["DATA"].asString(), X, y);

        auto lda = create_lda_for_train(args, X, y);

        // Add the listeners to be used
        if (!args["--quiet"].asBool()) {
            lda.get_event_dispatcher()->add_listener<EpochProgress>();
            lda.get_event_dispatcher()->add_listener<ExpectationProgress>();
            lda.get_event_dispatcher()->add_listener<MaximizationProgress>();
        }

        if (args["--snapshot_every"].asLong() > 0) {
            lda.get_event_dispatcher()->add_listener<SnapshotEvery>(
                args["MODEL"].asString(),
                args["--snapshot_every"].asLong()
            );
        }

        // Fit LDA model according to the given training data and parameters
        lda.fit(X, y);

        //Save the trained model
        io::save_lda(
            args["MODEL"].asString(),
            lda.model_parameters()
        );
    }
    else if (args["transform"].asBool()) {
        
        Eigen::MatrixXi X, y;
        // Parse data from input file
        io::parse_input_data(args["DATA"].asString(), X, y);

        // Load LDA model from file
        auto model = io::load_lda(args["MODEL"].asString());

        auto lda = create_lda_for_transform(args, model);

        // Add the listeners to be used
        if (!args["--quiet"].asBool()) {
            lda.get_event_dispatcher()->add_listener<EpochProgress>();
            lda.get_event_dispatcher()->add_listener<ExpectationProgress>();
        }

        Eigen::MatrixXd doc_topic_distribution;
        doc_topic_distribution = lda.transform(X);

        numpy_format::save(
            args["OUTPUT"].asString(),
            doc_topic_distribution
        );
    }
    else {
        std::cout << "Invalid command" << std::endl;
    }

    return 0;
}
