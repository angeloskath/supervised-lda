#ifndef _IESTEP_HPP_
#define _IESTEP_HPP_

#include <Eigen/Core>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/events/Events.hpp"
#include "ldaplusplus/Parameters.hpp"

using namespace Eigen;

namespace ldaplusplus {
namespace em {


/**
  * Interface that defines an E-step iteration for any LDA inference.
  *
  * The expectation step maximizes the likelihood (actually the Evidence Lower
  * Bound) of the data given constant parameters. In variational inference this
  * is achieved by changing the free variational parameters. In classical LDA
  * this step computes \f$\phi\f$ and \f$\gamma\f$ for every document given the
  * distribution over words for all topics, usually \f$\beta\f$ in literature.
  */
template <typename Scalar>
class IEStep : public events::EventDispatcherComposition
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:

        /**
          * Maximize the ELBO.
          *
          * @param doc        A single document
          * @param parameters An instance of class Parameters, which
          *                   contains all necessary model parameters 
          *                   for e-step's implementation
          * @return           The variational parameters for the current
          *                   model, after e-step is completed
          */
        virtual std::shared_ptr<Parameters> doc_e_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> parameters
        )=0;

        /**
         * Perform actions that should be performed once for each epoch for the
         * whole corpus. One use of this method is so that the e steps can know
         * which epoch they are running for.
         */
        virtual void e_step()=0;
};

}  // namespace em
}  // namespace ldaplusplus

#endif //  _IESTEP_HPP_
