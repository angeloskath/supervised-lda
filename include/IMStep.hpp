#ifndef _IMSTEP_HPP_
#define _IMSTEP_HPP_

#include <Eigen/Core>

#include "Document.hpp"
#include "Events.hpp"
#include "Parameters.hpp"

using namespace Eigen;

template <typename Scalar>
class IMStep : public EventDispatcherComposition
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:

        /**
         * Maximize the ELBO.
         *
         * @param model_parameters           Model parameters, after being updated in m_step
         */
        void void m_step(
            std::shared_ptr<Parameters> model_parameters
        )=0;

        /**
         * This function calculates all necessary parameters, that
         * will be used for the maximazation step.
         *
         * @param doc                        A single document
         * @param variational_parameters     The variational parameters used in m-step
         *                                   in order to maximize model parameters
         * @param model_parameters           Model parameters, used as output in case of 
         *                                   online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> variational_parameters,
            std::shared_ptr<Parameters> model_parameters
        )=0;
};

#endif  // _IMSTEP_HPP_
