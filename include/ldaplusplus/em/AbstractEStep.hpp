#ifndef _LDAPLUSPLUS_EM_ABSTRACTESTEP_HPP_
#define _LDAPLUSPLUS_EM_ABSTRACTESTEP_HPP_

#include <random>

#include "ldaplusplus/em/EStepInterface.hpp"
#include "ldaplusplus/utils.hpp"

namespace ldaplusplus {
namespace em {

/**
 * A base class to that provides few common functionalities for implementing an
 * E step.
 *
 * - Implements an empty void e_step() since most work happens in doc_e_step()
 * - Provides convergence check based on variational parameter \f$\gamma\f$
 * - Provides a PRNG initialized using a seed in the constructor
 */
template <typename Scalar>
class AbstractEStep : public EStepInterface<Scalar>
{
    typedef math_utils::ThreadSafePRNG<std::default_random_engine> PRNG;

    public:
        /**
         * Require a random state to be passed from the extending classes.
         *
         * @param random_state An initial seed value for random number
         *                     generation.
         */
        AbstractEStep(int random_state);

        /**
         * Implement an empty e_step because almost nobody needs to perform
         * some action at the end of each corpus epoch.
         */
        virtual void e_step() override {}

    protected:
        /**
         * Check for convergence based on the mean relative change of the
         * variational parameter \f$\gamma\f$.
         *
         * @gamma gamma_old The gamma of the previous iteration.
         * @gamma gamma     The gamma of this iteration.
         * @param tolerance The threshold below which we declare convergence.
         * @return Whether the change is small enough to indicate convergence.
         */
        bool converged(
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & gamma_old,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & gamma,
            Scalar tolerance
        );

        /**
         * Return a PRNG for use with any distribution.
         *
         * Although this isn't all that different from making random_ protected
         * it could allow for future change of the object returned since it can
         * be anything that satisfies the UniformRandomBitGenerator (see:
         * http://en.cppreference.com/w/cpp/concept/UniformRandomBitGenerator).
         */
        PRNG &get_prng() { return random_; }

    private:
        // A random number generator to be used for every random number needed
        // in this E step (initialized with random_state constructor parameter)
        PRNG random_;
};


}  // namespace em
}  // namespace ldaplusplus

#endif  // _LDAPLUSPLUS_EM_ABSTRACTESTEP_HPP_
