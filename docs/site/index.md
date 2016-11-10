# LDA++

LDA++ is a C++ library and a set of accompanying console applications that
enable the inference of various [Latent Dirichlet
Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) models.

Among the already implemented LDA variations are:

* [Unsupervised LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
* [Supervised LDA (sLDA)](http://www.cs.cmu.edu/~chongw/papers/WangBleiFeiFei2009.pdf)
* [Fast supervised LDA (fsLDA)](http://mug.ee.auth.gr/discovering-micro-events-video-data-using-topic-modeling/)

## Why LDA++?

**Modular architecture** allows the implementation of novel LDA variations with
minimal code.

**Clean implementations** enable a deep understanding of the variational
inference procedure followed for the available LDA models.

**Efficient multithreaded implementations** enable the inference of topics even
for large-scale datasets. Check our [research
page](http://mug.ee.auth.gr/discovering-micro-events-video-data-using-topic-modeling/)
for our new method to infer topics in a supervised manner, which is tested on
UCF-101 video dataset.

## Documentation

You can navigate the documentation from the top navigation bar but we also
provide a list of useful links below.

* [Installation instructions](/installation/)
* [Using the console applications](/console-applications/)
* [Visualization of topic inference](/topic-inference-visualization/)
* [Getting started with the LDA++ library](/getting-started/)
* [API Documentation](/api/html/)

## Example

In this section, we provide an example to point out how this library works. The
code below trains a fast supervised model (from our paper) using online
training and all the available threads. It infers 10 topics and runs 15
iterations. You can use the shell commands below to execute the example. We
also demonstrate the event system that is used to allow the models to report
back to your code asynchronously which we use to
report the likelihood and the progress.

If the example below seems too big or too complex you should check the [getting
started](/getting-started/) in the documentation as well as some of the other
tutorials. If you prefer using the console applications instead of writing code
then you should read their [documentation page](/console-applications/).

```cpp
#include <fstream>
#include <iostream>
#include <memory>

#include <ldaplusplus/LDA.hpp>
#include <ldaplusplus/LDABuilder.hpp>
#include <ldaplusplus/NumpyFormat.hpp>
#include <ldaplusplus/events/ProgressEvents.hpp>

using namespace ldaplusplus;

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Incorrect number of arguments." << std::endl;
        std::cout << "Usage: " << *argv << " [input_file] [output_file]"
                  << std::endl;

        return 1;
    }

    std::fstream input_file(argv[1], std::ios::in | std::ios::binary);
    std::fstream output_file(argv[2], std::ios::out | std::ios::binary);
    Eigen::MatrixXi X; // the documents
    Eigen::VectorXi y; // their classes

    // read the documents from a numpy formatted input file
    numpy_format::NumpyInput<int> ni;
    input_file >> ni; X = ni;
    input_file >> ni; y = ni;

    // all the parameters below are the default and can be omitted
    LDA<double> lda = LDABuilder<double>()
        .set_fast_supervised_e_step(
            10,   // expectation iterations
            1e-2, // expectation tolerance
            1,    // C parameter of fsLDA (see the paper)
            0.01, // percentage of documents to compute likelihood for
            42    // the randomness seed
        )
        .set_fast_supervised_online_m_step(
            10,   // number of classes in the dataset
            0.01, // the regularization penalty
            128,  // the minibatch size
            0.9,  // momentum for SGD training
            0.01, // learning rate for SGD
            0.9   // weight for the LDA natural gradient
        )
        .initialize_topics_seeded(
            X,  // the documents to seed from
            10, // the number of topics
            42  // the randomness seed
        )
        .initialize_eta_zeros(10) // initialize the supervised parameters
        .set_iterations(15);

    // add a listener to calculate and print the likelihood for every iteration
    // and a progress for every 128 documents (for every minibatch)
    double likelihood = 0;
    int count_likelihood = 0;
    int count = 0;
    lda.get_event_dispatcher()->add_listener(
        [&likelihood, &count, &count_likelihood](std::shared_ptr<events::Event> ev) {
            // an expectation has finished for a document
            if (ev->id() == "ExpectationProgressEvent") {
                count++; // seen another document
                if (count % 128 == 0) {
                    std::cout << count << std::endl;
                }

                // aggregate the likelihood if computed for this document
                auto expev =
                    std::static_pointer_cast<events::ExpectationProgressEvent<double> >(ev);
                if (expev->likelihood() < 0) {
                    likelihood += expev->likelihood();
                    count_likelihood ++;
                }
            }

            // A whole pass from the corpus has finished print the approximate per
            // document likelihood and reset the counters
            else if (ev->id() == "EpochProgressEvent") {
                std::cout << "Per document likelihood ~= "
                          << likelihood / count_likelihood << std::endl;
                likelihood = 0;
                count_likelihood = 0;
                count = 0;
            }
        }
    );

    // run the training for 15 iterations (we could also manually run each
    // iteration using partial_fit())
    lda.fit(X, y);

    // get the trained model and save it in numpy format
    auto model =
        std::static_pointer_cast<parameters::SupervisedModelParameters<double> >(
            lda.model_parameters()
        );

    // save matrices and vectors that can be loaded using numpy.load()
    output_file << numpy_format::NumpyOutput<double>(model->alpha);
    output_file << numpy_format::NumpyOutput<double>(model->beta);
    output_file << numpy_format::NumpyOutput<double>(model->eta);

    return 0;
}
```

Assuming you have already installed LDA++ on your system, simply copy and paste
the following instructions in a terminal to compile the previous example and
infer topics on the 60000 images from MNIST dataset.

```bash
$ wget "http://ldaplusplus.com/files/mnist.tar.gz"
$ tar -zxf mnist.tar.gz
$ g++ example.cpp -lldaplusplus -o example
$ ./example minst_train.npy model.npy
```

## Citation

Please cite our paper if it helped your research.

```
@inproceedings{Katharopoulos,
    author = {Katharopoulos, Angelos and Paschalidou, Despoina and Diou, Christos and Delopoulos, Anastasios},
    title = {Fast Supervised LDA for Discovering Micro-Events in Large-Scale Video Datasets},
    booktitle = {Proceedings of the 2016 ACM on Multimedia Conference},
    series = {MM '16},
    year = {2016},
    isbn = {978-1-4503-3603-1},
    location = {Amsterdam, The Netherlands},
    pages = {332--336},
    numpages = {5},
    url = {http://doi.acm.org/10.1145/2964284.2967237},
    doi = {10.1145/2964284.2967237},
    acmid = {2967237},
    publisher = {ACM},
    address = {New York, NY, USA},
    keywords = {supervised topic modeling, variational inference, video event detection, video micro-events},
} 
```

## License

LDA++ is released under the MIT license which practically allows anyone to do anything with it.
