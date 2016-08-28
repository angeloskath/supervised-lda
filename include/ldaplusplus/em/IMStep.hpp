#ifndef _IMSTEP_HPP_
#define _IMSTEP_HPP_

#include <Eigen/Core>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/events/Events.hpp"
#include "ldaplusplus/Parameters.hpp"

using namespace Eigen;

namespace ldaplusplus {
namespace em {


/**
 * Interface that defines an M-step iteration for any LDA inference.
 *
 * The maximization step maximizes the likelihood (actually the Evidence Lower
 * Bound) of the data by changing the parameters and using the variational
 * parameters as constants. In classical LDA this step computes the
 * distribution over words for all topics using the variational parameters
 * \f$\phi\f$ and \f$\gamma\f$.
 */
template <typename Scalar>
class IMStep : public events::EventDispatcherComposition
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:

        /**
         * Maximize the ELBO.
         *
         * This function usually changes the passed in parameters.
         *
         * @param parameters Model parameters (maybe changed after call)
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        )=0;

        /**
         * Perform calculations for a specific document.
         *
         * The variational parameters are only passed to the maximization step
         * in this method. In other implementations this method is usually
         * called *sufficient statistics*.
         *
         * This method allows for the implementation of online LDA inference
         * methods.
         *
         * @param doc          A single document
         * @param v_parameters The variational parameters computed in the e-step
         * @param m_parameters Model parameters could be changed in case of 
         *                     online methods
         */
        virtual void doc_m_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        )=0;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _IMSTEP_HPP_
