Console applications
====================

LDA++ provides a set of command-line executables for a corresponding set of LDA
variants in order to allow fast and easy experimentation, without the overhead
of writing C++ code. Below is the thorough list of console applications
implemented in LDA++:

- **lda** is the console application for Unsupervised LDA.
- **slda** is the console application for Supervised LDA (sLDA).
- **fslda** is the console application for Fast Unsupervised LDA (fsLDA).

At this point, it is important to note that the parsing of all command line
arguments, in all console applications, is done using
**[Docopt](https://github.com/docopt/docopt.cpp)**. Therefore, before building
them, make sure that **Docopt** is already installed in your system.


Basic commands
==============

All implemented console applications have two basic commands **train** and
**transform**. The first command is used to **train** a specific model (which
can be either unsupervised or supervised or fast supervised ) from a set of
input data, while the second one is used to **transform** a set of input data
according to an already trained model. Both commands have similar formats which
are presented below:

```bash
# Train command.
# The console_application_name can be either lda, slda or fslda according to
# the type of the LDA variant. The DATA refers to the path of the input data
# according to which we will train a model. The MODEL refers to the path where
# the trained model will be saved.
$ console_application_name train DATA MODEL

# Transform command.
# The console_application_name can be either lda, slda or fslda, according to
# the type of LDA variant. The MODEL refers to the path of the already trained
# model. The DATA refers to the path of the input data that will be transformed
# The OUTPUT refers to the path, where the transformed DATA will be saved.
$ console_application_name transform MODEL DATA OUTPUT

```

It can be easily seen that in case of the **train** command, the user has to specify two
paths, the first one corresponds to the input data, while the second one refers
to the path where the trained model will be saved. In order to make all these a
bit more clear, let us assume that we want to train a supervised LDA model and
the desired path to save the trained model is `/tmp/supervised_model`, while
the path to the input data is `/tmp/input_data`. In order to use the **slda**
console application to train a supervised LDA model with the aforementioned
paths, one should simply execute the following bash command.

```bash
$ ./slda train /tmp/input_data /tmp/supervised_model
E-M Iteration 1
100
200
...
```

On the other hand, in the case of the **transform** command, the user has to
specify three paths, the first one corresponds to the trained LDA model,
according to which we will transform the input data, the second one refers to
the path where the input data are stored, while the last one refers to the path
where the transformed data should be saved. Let us continue the previous
example, where we have already trained a supervised LDA model from a set of
input data using the **slda** console application. In order to transform these
data according to previously trained model and save the transformed results in
`/tmp/transformed_data`, one should simply execute the following bash command.

```bash
$ ./slda transform /tmp/supervised_model /tmp/input_data /tmp/transformed_data
E-M Iteration 1
100
200
...
```

Optional arguments
------------------

In the previous section, we discussed how one could use the provided console
applications to train a specific LDA variant from a set of input data. However,
apart from the **MODEL** and the **DATA** paths the user can provide additional
command-line arguments, which are common for every console application. All
these arguments are optionals, namely it is not necessary to assign a value to
the corresponding variable.

LDA++ implements a variational Expectation-Maximization (EM) procedure for the
parameter estimation. To be more precise, we perform variational inference for
learning the variational parameters in E-step, while we perform parameter
estimation in M-step. The number of Expectation-Maximization (EM) steps that
should be executed in order to train a model can be specified, by setting the
value of **iterations** argument. In addition, the user can specify the number
of jobs to be used in the Expectation step, by setting the value of **workers**
argument. The **snapshot_every** parameter is used to specify the number of EM
steps, after which a model will be saved in the defined path. For example, if
we set the **snapshot_every** argument to 5 and the **iterations** argument to
20, we will end up with 4 trained model, the first one will refer to the 5th
iteration, the second one to the 10th, the third one to the 15th and the last
one to 20th iteration. Moreover, the user can change the number of topics, by
setting the **topics** optional argument and the seed value, used for the
generation of random numbers, be setting the **random_state** argument. 

Finally, the last argument in all provided console applications is **continue**.
This argument is used to define the path of an already trained model from which
we want to continue training. Let us assume that we have already trained a
model for some iterations but the inferred topics are not good enough, so we
want to continue training for more iterations. To do that, we merely have to
set the **continue** argument to the path of the model to be further trained.

The following list summarizes all the optional arguments that can be
specified during the training process of each and every LDA variant implemented
in LDA++.

- **topics**: The number of topics used to train a specific LDA variant
  (default=100).
- **iterations**: The number of Expectation-Maximization steps used to train a
  model (default=20).
- **random_state**: Pseudo-random number generator seed control (default=0).
- **snapshot_every**: The number of iterations after which one model will be
  saved (default=-1).
- **workers**: The number of concurrent threads used during Expectation step
  (default=1).
- **continue**: A model to continue the training from

I/O format
===========

All implemented console applications expect and save data in **numpy format**.
The input data may contain either one or two numpy arrays of type **np.int32**,
depending on the type of console application. In case of **lda**, the input
data contain one array that holds the word counts of every document, thus its size is 
`(vocabulary_size, number_of_documents)`. On the contrary, in case
of **slda** and **fslda** the input data contain the aforementioned array
accompanied by second one which contains the corresponding class labels of each
document, thus it is of size `(number_of_documents,)`.

The following code snippet creates 100 random documents with a vocabulary of
1000 words and saves both them and their corresponding labels in numpy format.

```python
>>> import numpy as np

>>> X = np.round(np.maximum(0, np.log(np.random.rand(100, 100)+0.5))*20).astype(np.int32)

>>> y = np.round(np.random.rand(100)*3).astype(np.int32)
```

