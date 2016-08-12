# Welcome to LDA++

LDA++ is a C++ library for [Latent Dirichlet
Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf). The
primary goal of this library is to allow fast and easy experimentation with the
variational inference algorithm for inference of various LDA models.

Among the already implemented LDA variations are:

* LDA [[1]](#lda)
* Fast supervised LDA (fsLDA) [[2]](#fslda)
* Supervised LDA (sLDA) [[3]](#slda)

## Example

In the following section, we provide an example to point out how this library
works. In the current example we train an unsupervised LDA model on
artificially generated data.

``` cpp
#include <iostream>
#include <Eigen/Core>
#include <supervised-lda/LDABuilder.hpp>

using namespace Eigen;

int main(int argc, char ** argv) {
    // Generate into X 100 random documents with a vocabulary of 25 words
    ArrayXXd Xtmp = (ArrayXXd::Random(25, 100) + 1)/2 * 10;
    MatrixXi X = Xtmp.cast<int>().matrix();
    // Since the library was initially implementing just supervised variants of
    // LDA we need to create some random ys to accompany our words
    VectorXi y = VectorXi::Random(100);

     // Build our LDA
     LDABuilder<double> ldabuilder;
     ldabuilder.set_iterations(100)
               .set_workers(4)
               .set_e(ldabuilder.get_fast_classic_e_step())
               .set_batch_m_step()
               .initialize_topics("seeded", X);

      // Get an instance from the builder and train it
      LDA<double> lda = ldabuilder;
      lda.fit(X, y);

      // Print out the inferred topics
      std::cout << lda.model_parameters()->beta << std::endl;
}
```

We perform the same training using the console application distributed with the
library having generated the data in python and saved them in a file named
data.npy.

```bash
$ bin/slda train --topics 10 --iterations 100 --unsupervised_e_step --workers 4 data.npy model.npy
```

## Installation

The installation is not streamlined yet and it requires its own documentation
page. But long story short LDA++ still has a lot of dependencies a short list
of which follows:

* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [docopt.cpp](https://github.com/docopt/docopt.cpp)
* g++ 4.9
* googletest (for testing)

## License

LDA++ is released under the MIT license which practically allows anyone to do anything with it.

## References

<a name="lda"/>[[1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet
allocation." Journal of machine Learning research 3.Jan (2003):
993-1022.](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)</a>

<a name="fslda"/>[[2] Angelos Katharopoulos, Despoina Paschalidou, Christos Diou, Anastasios
Delopoulos. 2016. Fast Supervised LDA for Discovering Micro-Events in
Large-Scale Video Datasets. ACM International conference on Multimedia (MM
'16)](http://dx.doi.org/10.1145/2964284.2967237)</a>

<a name="slda"/>[[3] Chong, Wang, David Blei, and Fei-Fei Li. "Simultaneous image
classification and annotation." Computer Vision and Pattern Recognition, 2009.
CVPR 2009. IEEE Conference on. IEEE,
2009.](http://www.cs.cmu.edu/~chongw/papers/WangBleiFeiFei2009.pdf)</a>

