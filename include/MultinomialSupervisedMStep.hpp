#ifndef _MULTINOMIAL_SUPERVISED_M_STEP_HPP_
#define _MULTINOMIAL_SUPERVISED_M_STEP_HPP_

#include "IMStep.hpp"

namespace ldaplusplus {


template <typename Scalar>
class MultinomialSupervisedMStep : public IMStep<Scalar>
{
    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
    typedef Matrix<Scalar, Dynamic, 1> VectorX;
    
    public:
        MultinomialSupervisedMStep(Scalar mu = 2.)
            : mu_(mu)
        {}

        /**
         * @inheritdoc
         */
        virtual void m_step(
            std::shared_ptr<Parameters> parameters
        ) override;

        /**
         * @inheritdoc
         */
        virtual void doc_m_step(
            const std::shared_ptr<Document> doc,
            const std::shared_ptr<Parameters> v_parameters,
            std::shared_ptr<Parameters> m_parameters
        ) override;

    private:
        MatrixX phi_scaled_;
        VectorX phi_scaled_sum_;
        MatrixX b_;
        MatrixX h_;
        Scalar mu_;

        Scalar log_py_;
};
}
#endif  // _MULTINOMIAL_SUPERVISED_M_STEP_HPP_
