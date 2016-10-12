LDA++
=====

LDA++ is a C++ library and a set of accompanying console applications that
enable the inference of various [Latent Dirichlet
Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) models.

The project provides three [console applications](http://ldaplusplus.com/console-applications/)

* **lda** implementing **LDA**
* **slda** implementing **Supervised LDA**
* **fslda** implementing **Fast Supervised LDA**

and a library that can be used from your own C++ projects.

You can read the documentation site at
[ldaplusplus.com](http://ldaplusplus.com/) and there is of course an [API
documentation](http://ldaplusplus.com/api/) as well.

How to get it
-------------

We use CMake for building the project and currently only provide the option to
build from source. The [LDA++
installation](http://ldaplusplus.com/installation/) process is straightforward
and documented at our site.

Console applications
--------------------

We expect that the preferred way of using LDA++ will be through the provided
console applications. You can read thorough [documentation for
them](http://ldaplusplus.com/console-applications/) as well. All our console
applications are designed to read matrix files serialized in numpy format so
that one can easily create files in a python session.

It suffices to say that the following shell session runs the Fast Supervised
LDA (fsLDA) on the scikit learn digits dataset (provided you have installed
LDA++).

    $ python
    Python 2.7.12 (default, Jul  1 2016, 15:12:24) 
    [GCC 5.4.0 20160609] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from sklearn.datasets import load_digits
    >>> import numpy as np
    >>> d = load_digits()
    >>> with open("digits.npy", "wb") as f:
    ...     np.save(f, d.data.astype(np.int32).T)
    ...     np.save(f, d.target.astype(np.int32))
    ... 
    >>> exit()
    $ fslda train digits.npy model.npy
    E-M Iteration 1
    100
    200
    300
    400
    500
    600
    700
    800
    900
    1000
    1100
    1200
    1300
    1400
    1500
    1600
    1700
    log p(y | \bar{z}, eta): -4137.75
    log p(y | \bar{z}, eta): -3230.67
    log p(y | \bar{z}, eta): -2758.81
    log p(y | \bar{z}, eta): -2498.32
    log p(y | \bar{z}, eta): -2341.4
    log p(y | \bar{z}, eta): -2240.48
    log p(y | \bar{z}, eta): -2172.38
    log p(y | \bar{z}, eta): -2124.71
    log p(y | \bar{z}, eta): -2090.4
    log p(y | \bar{z}, eta): -2065.15
    ...
    $ python
    Python 2.7.12 (default, Jul  1 2016, 15:12:24) 
    [GCC 5.4.0 20160609] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numpy as np
    >>> with open("model.npy") as f:
    ...     alpha = np.load(f)
    ...     beta = np.load(f)
    ...     eta = np.load(f)
    ...
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(beta[0].reshape(8, 8), interpolation='nearest', cmap='gray')
    <matplotlib.image.AxesImage object at 0x7f4cf201b810>
    >>> plt.show()
    >>> exit()

License
-------

MIT license found in the LICENSE file.
