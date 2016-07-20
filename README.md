LDA++
=====

LDA++ was developed to allow quick experimentation with variational
inference methods for Latent Dirichlet Allocation based models.

The project is designed in two separate parts. A library that allows the
implementation of new variational inference methods for models based on LDA and
a console application that allows the use of most models (focusing on
supervised LDA) on serialized numpy arrays.

Dedicated documentation site coming soon.

Models implemented
------------------

* Unsupervised LDA
* Categorical supervised LDA
* Categorical fast approximate supervised LDA (paper coming soon...)
* Single tag correspondence LDA

Build
-----

After resolving the following dependencies

* [Eigen](http://eigen.tuxfamily.org/dox/)
* [docopt.cpp](https://github.com/docopt/docopt.cpp)
* [googletest](https://github.com/google/googletest) (platform wide install)

Also **g++4.9** is explicitly defined in the Makefile but this should change in
the future. For now just change it to whatever else you want but your mileage
may vary.

Then all you have to do is `make` and `make check`. To build a benchmark `make
bin/benchmark_name_without_extension` and then run the executable generated.

Console application
-------------------

The help message of the console application is the following

    Supervised LDA and other flavors of LDA.
    
        Usage:
            slda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                       [--m_step_iterations=MI] [--e_step_tolerance=ET]
                       [--m_step_tolerance=MT] [--fixed_point_iterations=FI]
                       [--multinomial] [--correspondence] [--mu=MU] [--eta_weight=EW]
                       [--unsupervised_e_step] [--fast_e_step]
                       [--second_order_m_step] [--online_m_step] [--semi_supervised]
                       [--supervised_weight=C] [--show_likelihood]
                       [--supervised_weight_evolution=WE] [--regularization_penalty=L]
                       [--beta_weight=BW] [--momentum=MM] [--learning_rate=LR]
                       [--batch_size=BS]
                       [-q | --quiet] [--snapshot_every=N] [--workers=W]
                       [--continue=M] DATA MODEL
            slda transform [-q | --quiet] [--e_step_iterations=EI]
                           [--e_step_tolerance=ET] [--workers=W]
                           MODEL DATA OUTPUT
            slda evaluate [-q | --quiet] [--e_step_iterations=EI]
                          [--e_step_tolerance=ET] [--workers=W]
                          MODEL DATA
            slda (-h | --help)
    
        Options:
            -h, --help         Show this help
            -q, --quiet         Produce no output to the terminal
            --topics=K          How many topics to train [default: 100]
            --iterations=I      Run LDA for I iterations [default: 20]
            --e_step_iterations=EI  The maximum number of iterations to perform
                                    in the E step [default: 10]
            --e_step_tolerance=ET   The minimum accepted relative increase in log
                                    likelihood during the E step [default: 1e-4]
            --unsupervised_e_step   Use the unsupervised E step to calculate phi and gamma
            --fast_e_step           Choose a variant of E step that doesn't compute
                                    likelihood in order to be faster
            --second_order_m_step   Use the second order approximation for the M step
            --online_m_step         Choose online M step that updates the model
                                    parameters after seeing mini_batch documents
            --semi_supervised       Train a semi supervised lda
            -C C, --supervised_weight=C   The weight of the supervised term for the
                                          E step [default: 1]
            --show_likelihood       Compute supervised likelihood during the e step
            --supervised_weight_evolution=WE    Choose the weight evolution
                                                strategy for the supervised part of
                                                the e step [default: constant]
            --m_step_iterations=MI  The maximum number of iterations to perform
                                    in the M step [default: 200]
            --m_step_tolerance=MT   The minimum accepted relative increase in log
                                    likelihood during the M step [default: 1e-4]
            --fixed_point_iterations=FI  The number of fixed point iterations to compute
                                         \phi [default: 20]
            --multinomial           Use the multinomial version of supervised LDA
            --correspondence        Use the correspondence version of supervised LDA
            --mu=MU                 The multinomial prior on the naive bayesian
                                    classification [default: 2]
            --eta_weight=EW         The weight of eta in the multinomial phi update [default: 1]
            -L L, --regularization_penalty=L  The regularization penalty for the Multinomial
                                              Logistic Regression [default: 0.05]
            --beta_weight=BW        Set the weight of the previous beta parameters
                                    w.r.t to the new from the minibatch [default: 0.9]
            --momentum=MM           Set the momentum for changing eta [default: 0.9]
            --learning_rate=LR      Set the learning rate for changing eta [default: 0.01]
            --batch_size=BS         The mini-batch size for the online learning [default: 128]
            --snapshot_every=N      Snapshot the model every N iterations [default: -1]
            --workers=N             The number of concurrent workers [default: 1]
            --continue=M            A model to continue training from

We can create a very small example dataset in 6 lines of python code and then
run the console application to infer topics on it.

    # We will be needing that since the LDA++ console app
    # reads serialized numpy arrays.
    import numpy as np
    
    # Create 100 random documents with a vocabulary of 100 words. The cast to
    # int32 at the end is mandatory because the application expects that type
    # of array.
    X = np.round(np.maximum(0, np.log(np.random.rand(100, 100)+0.5))*20).astype(np.int32)

    # Choose labels for our 100 documents.
    y = np.round(np.random.rand(100)*3).astype(np.int32)

    # Finally save the data in a file to be passed to the console application.
    with open("/tmp/data.npy", "wb") as data:
        np.save(data, X)
        np.save(data, y)

After running the script above we can run a multitude of LDA methods on our data.

    $ bin/slda train --topics 4 --unsupervised_e_step /tmp/data.npy /tmp/model
    ...
    ...
    ...
    $ bin/slda train --topics 4 --fast_e_step -C 0.1 --show_likelihood /tmp/data.npy /tmp/model
    ...
    ...
    ...
    $ bin/slda train --topics 4 --show_likelihood /tmp/data.npy /tmp/model
    ...
    ...
    ...
    $ bin/slda evaluate /tmp/model /tmp/data.npy
    ...
    ...
    ...


License
-------

MIT license found in the LICENSE file.
