#ifndef _LDA_BUILDER_HPP_
#define _LDA_BUILDER_HPP_


#include <memory>
#include <stdexcept>
#include <thread>

#include <Eigen/Core>

#include "initialize.hpp"
#include "ApproximatedSupervisedEStep.hpp"
#include "FastUnsupervisedEStep.hpp"
#include "IEStep.hpp"
#include "IMStep.hpp"
#include "LDA.hpp"
#include "OnlineSupervisedMStep.hpp"
#include "SemiSupervisedEStep.hpp"
#include "SemiSupervisedMStep.hpp"
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
              workers_(std::thread::hardware_concurrency()),
              e_step_(std::make_shared<UnsupervisedEStep<Scalar> >()),
              m_step_(std::make_shared<UnsupervisedMStep<Scalar> >()),
              model_parameters_(
                std::make_shared<SupervisedModelParameters<Scalar> >()
              )
        {}

        // set generic parameters
        LDABuilder & set_iterations(size_t iterations) {
            iterations_ = iterations;

            return *this;
        }
        LDABuilder & set_workers(size_t workers) {
            workers_ = workers;

            return *this;
        }

        // create e steps
        template <typename ...Args>
        std::shared_ptr<IEStep<Scalar> > get_classic_e_step(Args... args) {
            return std::make_shared<UnsupervisedEStep<Scalar> >(args...);
        }
        template <typename ...Args>
        std::shared_ptr<IEStep<Scalar> > get_fast_classic_e_step(Args... args) {
            return std::make_shared<FastUnsupervisedEStep<Scalar> >(args...);
        }
        template <typename ...Args>
        std::shared_ptr<IEStep<Scalar> > get_supervised_e_step(Args... args) {
            return std::make_shared<SupervisedEStep<Scalar> >(args...);
        }
        template <typename ...Args>
        std::shared_ptr<IEStep<Scalar> > get_fast_supervised_e_step(Args... args) {
            return std::make_shared<ApproximatedSupervisedEStep<Scalar> >(args...);
        }
        template <typename ...Args>
        std::shared_ptr<IEStep<Scalar> > get_semi_supervised_e_step(Args... args) {
            return std::make_shared<SemiSupervisedEStep<Scalar> >(args...);
        }
        LDABuilder & set_e(std::shared_ptr<IEStep<Scalar> > e_step) {
            e_step_ = e_step;
            return *this;
        }

        // create m steps
        template <typename ...Args>
        LDABuilder & set_batch_m_step(Args... args) {
            m_step_ = std::make_shared<UnsupervisedMStep<Scalar> >(args...);

            return *this;
        }
        template <typename ...Args>
        LDABuilder & set_supervised_batch_m_step(Args... args) {
            m_step_ = std::make_shared<SupervisedMStep<Scalar> >(args...);

            return *this;
        }
        template <typename ...Args>
        LDABuilder & set_supervised_online_m_step(Args... args) {
            m_step_ = std::make_shared<OnlineSupervisedMStep<Scalar> >(args...);

            return *this;
        }
        template <typename ...Args>
        LDABuilder & set_semi_supervised_batch_m_step(Args... args) {
            m_step_ = std::make_shared<SemiSupervisedMStep<Scalar> >(args...);

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

            if (
                model_parameters_->beta.rows() != model_parameters_->eta.rows() &&
                model_parameters_->eta.rows() > 0
            ) {
                throw std::runtime_error("\\eta and \\beta should be "
                                         "initialized with the same number of "
                                         "topics");
            }

            return LDA<Scalar>(
                model_parameters_,
                e_step_,
                m_step_,
                iterations_,
                workers_
            );
        };

    private:
        // generic lda parameters
        size_t iterations_;
        size_t workers_;

        // implementations
        std::shared_ptr<IEStep<Scalar> > e_step_;
        std::shared_ptr<IMStep<Scalar> > m_step_;

        // the model parameters
        std::shared_ptr<SupervisedModelParameters<Scalar> > model_parameters_;
};


#endif  //_LDA_BUILDER_HPP_
