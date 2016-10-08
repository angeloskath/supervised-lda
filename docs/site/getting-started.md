# Getting Started

The goal of this page is to introduce you to the C++ library **LDA++**.
If instead you just want to use LDA++ to infer topics from a corpus see
the [console applications](/console-applications/).

LDA++ has only one dependency to external libraries. It depends on
[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for efficient
matrix and vector operations. As it will be seen in the following sections
Eigen matrices and vectors are used to interface easily with the library. The
model parameters, for instance, are Eigen matrices and they can be printed with
`std::cout` or otherwise manipulated.

All of the classes in the library are parameterized using templates with
respect to the floating point scalar type. This allows us to save memory (and
maybe speed up) using single precision floats by changing a simple `double` to
`float`.

## LDA facade

All types of LDA training take place through the
[ldaplusplus::LDA](/api/html/classldaplusplus_1_1LDA.html) facade. This class
combines an **expectation step** implementation, **a maximization step**
implementation and some **model parameters** to perform variational inference
and compute the optimal LDA model parameters. The interface of LDA is heavily
inspired (the same really) with the Estimator, Transformer and Classifier
scikit-learn interfaces.

```cpp
namespace ldaplusplus {

template <typename Scalar>
class LDA
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;

    public:
        void fit(const Eigen::MatrixXi &X, const Eigen::VectorXi &y);
        void fit(const Eigen::MatrixXi &X);

        void partial_fit(const Eigen::MatrixXi &X, const Eigen::VectorXi &y);

        MatrixX transform(const Eigen::MatrixXi &X);

        MatrixX decision_function(const Eigen::MatrixXi &X);
        MatrixX decision_function(const MatrixX &Z);

        Eigen::VectorXi predict(const MatrixX &scores);
        Eigen::VectorXi predict(const Eigen::MatrixXi &X);

        const std::shared_ptr<parameters::Parameters> model_parameters();
        ...
}

}  // namespace ldaplusplus
```

The easiest way to interact with
[ldaplusplus::LDA](/api/html/classldaplusplus_1_1LDA.html) is through Eigen
matrices. As is expected **LDA::fit()** fits the model to the provided training
data mutating the model parameters which are accesible through
**LDA::model_parameters()**. The functions **LDA::decision_function()** and
**LDA::predict()** both assume supervised LDA with linear classifier.

Assuming we have created an LDA instance the following example showcases the
use of the facade.

```cpp
using namespace Eigen;

LDA<double> lda = ... get an LDA instance ...;

// Create 1000 random documents
MatrixXi X = (ArrayXXd::Random(100, 1000).abs() * 20).matrix().cast<int>();
VectorXi y = (ArrayXd::Random(1000).abs() * 5).matrix().cast<int>();

// Assuming lda represents a supervised model
lda.fit(X, y);
auto model = std:static_pointer_cast<parameters::SupervisedModelParameters<double> >(
    lda.model_parameters()
);
// Contains the dirichlet prior
VectorXd alpha = model->alpha;
// Contains the topics
MatrixXd beta = model->beta;
// Contains the supervised parameters eta
MatrixXd eta = model->eta;

// Create 100 random test documents
MatrixXi X_test = (ArrayXXd::Random(100, 100).abs() * 20).matrix().cast<int>();
// Z now contains the topic mixtures for each document
MatrixXd Z = lda.transform(X_test):
// y_test contains the predictions and in pseudocode is
// y_test = (lda.transform(X_test).transpose() * lda.model_parameters()->eta).argmax(axis=1)
VectorXi y_test = lda.predict(X_test);

// You can further train one more iteration using partial_fit
lda.partial_fit(X, y)
```

## LDABuilder

Although we could build an LDA instance directly using its constructor, it is
easier to use the provided builder
[ldaplusplus::LDABuilder](/api/html/classldaplusplus_1_1LDABuilder.html) to
ensure the readability of our code. A builder instance can be implicitly casted
to an LDA instance, thus the creation of a new LDA instance is as easy as the
following code.

```cpp
// Create an unsupervised lda with 10 topics expecting 1000 words vocabulary
LDA<double> lda = LDABuilder<double>().initialize_topics_uniform(1000, 10);
```

### Initialize model parameters

In order to create an LDA from an LDABuilder at least the model parameters must
be initialized. The LDABuilder checks if the model parameters have been
initialized correctly and throws a `std::runtime_error` in case they haven't.
In case of unsupervised LDA (which is the default for LDABuilder) only the
topics must be initialized using one of the `LDABuilder::initialize_topics_*()`
functions.

```cpp
namespace ldaplusplus {

template <typename Scalar>
class LDABuilder
{
    public:
        ...
        LDABuilder & initialize_topics_seeded(const Eigen::MatrixXi &X, size_t topics, ...);
        LDABuilder & initialize_topics_uniform(size_t words, size_t topics);
        LDABuilder & initialize_topics_from_model(
            std::shared_ptr<parameters::ModelParameters<Scalar> > model);
        ...
}  // namespace ldaplusplus
```

In the case of supervised topic models (sLDA and fsLDA) one must also
initialize the supervised model parameters (after initializing the topics)
using one of the `LDABuilder::initialize_eta_*()` functions.

```cpp
namespace ldaplusplus {

template <typename Scalar>
class LDABuilder
{
    public:
        ...
        LDABuilder & initialize_eta_zeros(size_t num_classes);
        LDABuilder & initialize_eta_uniform(size_t num_classes);
        LDABuilder & initialize_eta_from_model(
            std::shared_ptr<parameters::SupervisedModelParameters<Scalar> > model);
        ...
}  // namespace ldaplusplus
```

### Choose LDA method

Choosing the LDA method means choosing the variational inference method for
solving an LDA problem, namely the *Expectation Step* and the *Maximization
Step*. Unlike the [console applications](/console-applications/) which focus on
three models, the library contains a lot more models and allows users to define
their own. The LDABuilder has support for creating LDA models that use any of
the variational implementations that ship with the library.

Choosing an implementation for the Expectation and Maximization steps is done
by calling methods named `set_[method_name]_[e or m]_step`. Almost all methods
have sensible default parameters and we encourage you to read the Api
documentation of
[ldaplusplus::LDABuilder](/api/html/classldaplusplus_1_1LDA.html) for the
documentation of the parameters. For every `set_*` method there exists a
corresponding `get_*` method that returns a pointer to the corresponding
implementation instance. Next follows a list with all the available method
names:

* classic (LDA)
* supervised (sLDA)
* fast_supervised (fsLDA)
* fast_supervised_online (fsLDA online maximization step only)
* semi_supervised (experimental)
* multinomial_supervised (experimental)
* correspondence_supervised (experimental)

## Examples

In this section we will provide some examples using the LDABuilder to
instantiate various kinds of LDA models and use them on a small randomly
generated dataset.

The code below can be compiled, provided you have [installed](/installation/)
LDA++, with the following simple command `g++ -std=c++11 test.cpp -o test
-lldaplusplus`.

```cpp
#include <iostream>

#include <Eigen/Core>
#include <ldaplusplus/LDABuilder.hpp>

using namespace Eigen;
using namespace ldaplusplus;

int main() {
    // Define some variables that we will be using in LDA creation
    size_t num_classes = 5;
    size_t num_topics = 10;

    // Create a random dataset 100 words 50 documents and
    // corresponding class labels
    MatrixXi X = (ArrayXXd::Random(100, 50).abs() * 20).matrix().cast<int>();
    VectorXi y = (ArrayXd::Random(50).abs() * num_classes).matrix().cast<int>();

    // Create the simplest lda possible an Unsupervised LDA with uniform topic
    // initialization
    LDA<double> lda = LDABuilder<double>().initialize_topics_uniform(
        X.rows(),   // X.rows() is the number of words in the vocab
        num_topics  // how many topics we want to infer
    );

    // Create a supervised LDA as defined by Wang et al in Simultaneous image
    // classification and annotation
    LDA<double> slda = LDABuilder<double>()
                            .set_supervised_e_step()
                            .set_supervised_m_step()
                            .initialize_topics_seeded(X, num_topics)
                            .initialize_eta_zeros(num_classes); // we need to
                                                                // initialize eta
                                                                // as well now

    // Create a fast supervised LDA as defined in Fast Supervised LDA for
    // Discovering Micro-Events in Large-Scale Video Datasets
    LDA<double> fslda = LDABuilder<double>()
                            .set_fast_supervised_e_step()
                            .set_fast_supervised_m_step()
                            .initialize_topics_seeded(X, num_topics)
                            .initialize_eta_zeros(num_classes);

    // Train all our models
    lda.fit(X);
    slda.fit(X, y);
    fslda.fit(X, y);

    // Extract the top words of the unsupervised model
    auto model = std::static_pointer_cast<parameters::ModelParameters<double> >(
        lda.model_parameters()
    );
    VectorXi top_words(model->beta.rows());
    for (int i=0; i<model->beta.rows(); i++) {
        model->beta.row(i).maxCoeff(&top_words[i]);
    }
    std::cout << "Top Words:" << std::endl << top_words
              << std::endl << std::endl;

    // Now to transform the data using the slda model we need an unsupervised
    // lda model (because we do not know the class labels for the untransformed
    // data)
    LDA<double> transformer = LDABuilder<double>().initialize_topics_from_model(
        std::static_pointer_cast<parameters::ModelParameters<double> >(
            slda.model_parameters()
        )
    );
    MatrixXd Z = transformer.transform(X);
    std::cout << "The topic mixtures for the first document" << std::endl
              << Z.col(0) << std::endl << std::endl;

    
    // Predict the class labels using the fslda model (again we will be using
    // an unsupervised model because we do not know the class labels
    // beforehand)
    auto sup_model = std::static_pointer_cast<parameters::SupervisedModelParameters<double> >(
        fslda.model_parameters()
    );
    LDA<double> predictor = LDABuilder<double>()
            .initialize_topics_from_model(sup_model)
            .initialize_eta_from_model(sup_model);

    VectorXi y_pred = predictor.predict(X);
    std::cout << "Accuracy: " << (y.array() == y_pred.array()).cast<float>().mean()
              << std::endl;

    return 0;
}
```

And here follows a possible output

```
Top Words:
65
65
65
65
65
65
65
65
65
65

The topic mixtures for the first document
39.4252
64.1777
87.8473
 171.12
69.4335
85.5534
95.9499
66.3159
214.033
54.1435

Accuracy: 0.92
```
