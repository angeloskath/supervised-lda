#ifndef _LDA_BUILDER_HPP_
#define _LDA_BUILDER_HPP_


#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "IEStep.hpp"
#include "IMStep.hpp"
#include "LDA.hpp"
#include "SupervisedEStep.hpp"
#include "SupervisedMStep.hpp"
#include "UnsupervisedEStep.hpp"
#include "UnsupervisedMStep.hpp"


template <typename Scalar>
class ILDABuilder
{
    public:
        virtual operator LDA<Scalar>() const = 0;
};


/**
 * The LDABuilder provides a simpler interface to build an LDA.
 *
 * Examples:
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      initialize_topics("random", X, 100);
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      set_iterations(20).
 *                      set_e_step("classic").
 *                      set_m_step("supervised-batch").
 *                      initialize_topics("seeded", X, 100).
 *                      initialize_eta("zeros", X, y);
 *
 * LDA<double> lda = LDABuilder<double>().
 *                      set_e_step("classic").
 *                      set_m_step("supervised-batch").
 *                      initialize_topics_from_model(model).
 *                      initialize_eta_from_model(model);
 */
template <typename Scalar>
class LDABuilder : public ILDABuilder<Scalar>
{
    public:
        LDABuilder()
            : iterations_(20),
              e_step_(std::make_shared<UnsupervisedEStep<Scalar> >()),
              m_step_(std::make_shared<UnsupervisedMStep<Scalar> >()),
              model_parameters_(
                std::make_shared<SupervisedModelParameters<Scalar> >()
              )
        {}

        // set generic parameters
        LDABuilder & set_iterations(size_t topics);

        // create e steps
        template <typename ...Args>
        LDABuilder & set_e_step(const std::string &type, Args... args) {
            if (type == "classic") {
                e_step_ = std::make_shared<UnsupervisedEStep<Scalar> >(args...);
            }
            else if (type == "supervised") {
                e_step_ = std::make_shared<SupervisedEStep<Scalar> >(args...);
            }
            else {
                throw std::invalid_argument(type + " is an unknown expectation step");
            }

            return *this;
        }

        // create m steps
        template <typename ...Args>
        LDABuilder & set_m_step(const std::string &type, Args... args) {
            if (type == "batch") {
                m_step_ = std::make_shared<UnsupervisedMStep<Scalar> >(args...);
            }
            else if (type == "supervised-batch") {
                m_step_ = std::make_shared<SupervisedMStep<Scalar> >(args...);
            }
            else {
                throw std::invalid_argument(type + " is an unknown maximization step");
            }

            return *this;
        }

        // initialize model parameters
        template <typename ...Args>
        LDABuilder & initialize_topics(
            const std::string &type,
            const MatrixXi &X,
            Args... args
        ) {
            auto corpus = std::make_shared<EigenCorpus>(X);

            if (type == "seeded") {
                initialize_topics_seeded<Scalar>(model_parameters_, corpus, args...);
            }
            else if (type == "random") {
                initialize_topics_random<Scalar>(model_parameters_, corpus, args...);
            }
            else if (type == "model") {
                initialize_topics_from_model<Scalar>(
                    std::static_pointer_cast<ModelParameters<Scalar> >(args...)
                );
            }
            else {
                throw std::invalid_argument(type + " is an unknown topic initialization method");
            }

            return *this;
        }

        template <typename ...Args>
        LDABuilder & initialize_eta(
            const std::string &type,
            const MatrixXi &X,
            const VectorXi &y,
            Args... args
        ) {
            auto corpus = std::make_shared<EigenClassificationCorpus>(X, y);

            if (type == "zeros") {
                initialize_eta_zeros<Scalar>(model_parameters_, corpus, args...);
            }
            else if (type == "model") {
                initialize_eta_from_model<Scalar>(
                    std::static_pointer_cast<SupervisedModelParameters<Scalar> >(args...)
                );
            }
            else {
                throw std::invalid_argument(type + " is an unknown eta initialization method");
            }

            return *this;
        }

        LDABuilder & initialize_topics_from_model(
            std::shared_ptr<ModelParameters<Scalar> > model
        ) {
            model_parameters_->alpha = model->alpha;
            model_parameters_->beta = model->beta;

            return *this;
        }

        LDABuilder & initialize_eta_from_model(
            std::shared_ptr<SupervisedModelParameters<Scalar> > model
        ) {
            model_parameters_->eta = model->eta;

            return *this;
        }

        virtual operator LDA<Scalar>() const override {
            if (model_parameters_->beta.rows() == 0) {
                throw std::runtime_error("You need to call initialize_topics before "
                                         "creating an LDA from the builder.");
            }

            return LDA<Scalar>(
                model_parameters_,
                e_step_,
                m_step_,
                iterations_
            );
        };

    private:
        // generic lda parameters
        size_t iterations_;

        // implementations
        std::shared_ptr<IEStep<Scalar> > e_step_;
        std::shared_ptr<IMStep<Scalar> > m_step_;

        // the model parameters
        std::shared_ptr<SupervisedModelParameters<Scalar> > model_parameters_;
};


#endif  //_LDA_BUILDER_HPP_
