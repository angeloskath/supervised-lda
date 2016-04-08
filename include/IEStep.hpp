#ifndef _IESTEP_HPP_
#define _IESTEP_HPP_

#include <Eigen/Core>

#include "Document.hpp"
#include "Events.hpp"
#include "Parameters.hpp"

using namespace Eigen;

/**
  * Interface that implements an e-step iteration for a single document
  */
template <typename Scalar>
class IEStep : public EventDispatcherComposition
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:

        /**
          * Maximize the ELBO
          *
          * @param doc          A sinle document
          * @param parameters   An instance of class Parameters, which
          *                     contains all necessary model parameters 
          *                     for e-step's implementation
          * @return             The variational parameters for the current
          *                     model, after e-step is completed
          */
        virtual std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        )=0;

        /**
         * Perform action that should be performed once for each epoch for the
         * whole corpus. One use of this method is so that the e steps can know
         * which epcoh they are running for.
         *
         * TODO: Determine if any parameters are needed
         */
        virtual void e_step()=0;
};
#endif //  _IESTEP_HPP_
