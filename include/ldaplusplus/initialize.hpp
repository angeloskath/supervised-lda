#ifndef INITIALIZE_HPP
#define INITIALIZE_HPP

#include <Eigen/Core>

#include "ldaplusplus/Document.hpp"
#include "ldaplusplus/Parameters.hpp"

namespace ldaplusplus {


template<typename Scalar>
void initialize_topics_seeded(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<corpus::Corpus> corpus,
    size_t topics=600,
    int random_state=0
);

template <typename Scalar>
void initialize_topics_random(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<corpus::Corpus> corpus,
    size_t topics=600,
    int random_state=0
);

template <typename Scalar>
void initialize_eta_zeros(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<corpus::Corpus> corpus,
    size_t topics=600
);

template <typename Scalar>
void initialize_eta_multinomial(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<corpus::Corpus> corpus,
    size_t topics=600
);

}
#endif  // INITIALIZE_HPP
