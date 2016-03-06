#ifndef INITIALIZE_HPP
#define INITIALIZE_HPP

#include <Eigen/Core>

#include "Document.hpp"
#include "Parametes.hpp"

typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
typedef Matrix<Scalar, Dynamic, 1> VectorX;

template<typename Scalar>
void initialize_topics_seeded(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics=600,
    int random_state=0
);

template <typename Scalar>
void initialize_topics_random(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics=600
);

template <typename Scalar>
void initialize_eta_zeros(
    const std::shared_ptr<Parameters> parameters,
    const std::shared_ptr<Corpus> corpus,
    size_t topics
);

#endif  // INITIALIZE_HPP
