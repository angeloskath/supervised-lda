#ifndef _ESTEPINTERFACE_HPP_
#define _ESTEPINTERFACE_HPP_

#include <Eigen/Core>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/events/Events.hpp"
#include "ldaplusplus/Parameters.hpp"

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
class EStepInterface : public events::EventDispatcherComposition
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
    
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
        virtual std::shared_ptr<parameters::Parameters> doc_e_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> parameters
        )=0;

        /**
         * Perform actions that should be performed once for each epoch for the
         * whole corpus. One use of this method is so that the e steps can know
         * which epoch they are running for.
         */
        virtual void e_step()=0;

        virtual ~EStepInterface(){};
};

}  // namespace em
}  // namespace ldaplusplus

#endif //  _ESTEPINTERFACE_HPP_
