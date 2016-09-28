#ifndef _LDAPLUSPLUS_EM_MULTINOMIALSUPERVISEDMSTEP_HPP_
#define _LDAPLUSPLUS_EM_MULTINOMIALSUPERVISEDMSTEP_HPP_

#include "ldaplusplus/em/MStepInterface.hpp"

namespace ldaplusplus {
namespace em {


template <typename Scalar>
class MultinomialSupervisedMStep : public MStepInterface<Scalar>
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
    
    public:
        MultinomialSupervisedMStep(Scalar mu = 2.)
            : mu_(mu)
        {}

        /**
         * @inheritdoc
         */
        virtual void m_step(
            std::shared_ptr<parameters::Parameters> parameters
        ) override;

        /**
         * @inheritdoc
         */
        virtual void doc_m_step(
            const std::shared_ptr<corpus::Document> doc,
            const std::shared_ptr<parameters::Parameters> v_parameters,
            std::shared_ptr<parameters::Parameters> m_parameters
        ) override;

    private:
        MatrixX phi_scaled_;
        VectorX phi_scaled_sum_;
        MatrixX b_;
        MatrixX h_;
        Scalar mu_;

        Scalar log_py_;
};

}  // namespace em
}  // namespace ldaplusplus

#endif  // _LDAPLUSPLUS_EM_MULTINOMIALSUPERVISEDMSTEP_HPP_
