Console applications
====================

LDA++ provides a set of command-line executables for a set of corresponding LDA
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
**transform**. Both commands have similar formats that are presented below.

```bash
# Train command.
# The app_name can be either lda, slda or fslda according to the type of the
# LDA variant. The DATA refers to the path of the input data according to which
# we will train a model. The MODEL refers to the path where the trained model
# will be saved.
$ app_name train DATA MODEL

# Transform command.
# The app_name can be either lda, slda or fslda, according to the type of LDA
# variant. The MODEL refers to the path of the already trained model. The DATA
# refers to the path of the input data that will be transformed The OUTPUT
# refers to the path, where the transformed DATA will be saved.
$ app_name transform MODEL DATA OUTPUT

```

In case of the **train** command, the user has to specify two paths. The first
one corresponds to the input data, while the second one refers to the path
where the trained model will be saved. For example, if we want to train a
vanilla LDA according to a set of data stored in `/tmp/input_data` and save the
trained model to `/tmp/lda_model`, we should simple execute the following bash
command.

```bash
$ lda train /tmp/input_data /tmp/lda_model
E-M Iteration 1
100
200
...
```

On the other hand, in the case of the **transform** command, the user has to
specify three paths. The first one corresponds to the trained LDA model,
according to which we will transform the input data, the second one refers to
the path where the input data are stored, while the last one refers to the path
where the transformed data will be saved. If we continue the previous example,
we can now transform our data and save the transformed results in
`/tmp/transformed_data`, by simply executing the following bash command.

```bash
$ lda transform /tmp/lda_model /tmp/input_data /tmp/transformed_data
E-M Iteration 1
100
200
...
```

Optional arguments
------------------

Apart from the **MODEL** and the **DATA** paths the user can provide additional
command-line arguments, which are common for every console application.

LDA++ implements a [variational Expectation-Maximization (EM)
procedure](https://arxiv.org/pdf/1601.00670.pdf) for the parameter estimation.
To be more precise, we perform variational inference for learning the
variational parameters in E-step and parameter estimation in M-step. The number
of Expectation-Maximization steps that should be executed in order to train a
model can be specified, by setting the **iterations** argument. In addition,
the user can specify the number of threads to be used in the Expectation step,
by setting the value of the **workers** argument. The **snapshot_every**
parameter is used to specify the number of EM steps, after which a model will
be saved in the defined path. For example, if we set the **snapshot_every**
argument to 5 and the **iterations** argument to 20, we will end up with 4
trained model, the first one will refer to the 5th iteration, the second one to
the 10th, the third one to the 15th and the last one to 20th iteration.
Moreover, the user can change the number of topics, by setting the **topics**
optional argument and the seed value, used for the generation of random
numbers, by setting the **random_state** argument. 

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

- **initialize_random**: With this option, the topic over words distribution,
  namely $\beta$, are initialized as random distributions. **The default
  initialization method is initialize_seeded.**

- **initialize_seeded**: This option, allows user to initialize the topic over
  words distribution, namely $\beta$, by seeding them from a set of documents.
  **The default initialization method is initialize_seeded.**

I/O format
===========

All implemented console applications expect and save data in **numpy format**.
In the case of **lda**, which is the unsupervised variant of LDA, the input
data contain one array, let it be $X$, that holds the word counts of every
document in the corpus. On the contrary, in the case of **slda** and **fslda**,
which are supervised variants of LDA, the input data contain one additional
array, let it be $y$, that holds the corresponding class labels of each
document.  Consequently, $X$ is an array of size `(vocabulary_size,
number_of_documents)` and $y$ is an array of size `(number_of_documents,)`.
Both arrays should be of type **int32**, because all console applications
expect this specific type.

In the following python session we create an artificial dataset of 100
documents with a vocabulary of 1000 words. We assume that each document belongs
to one of the 6 classes. At the end of the example we save both $X$ and $y$ to
a binary file in numpy format.

```python
In [1]: import numpy as np

# Create 100 random documents with a vocabulary of 1000 words. The cast to
# int32 at the end is mandatory because all console applications expect that
# type of array
In [2]: X = np.round(np.maximum(0, np.random.rand(1000, 100)*20)).astype(np.int32)

# Create the class labels for all documents in the corpus. We multiply with 5,
# because we assume that there are 6 classes
In [3]: y = np.round(np.random.rand(100)*5).astype(np.int32)

# Save the data in a file that can be then passed to all console applications
In [4]: with open ("/tmp/data.npy", "wb") as data:
   ...:     np.save(data, X)
   ...:     np.save(data, y)
```

As soon as, we have saved our data in numpy format we can use them with all
implemented console applications. In the following bash command, we use the
newly created data and train a Fast Supervised LDA model with 10 topics. The
created model will be saved in numpy format.

```bash
$ fslda train /tmp/data.npy /tmp/fslda_model.npy --topics 10
E-M Iteration 1
100
log p(y | \bar{z}, eta): -179.176
log p(y | \bar{z}, eta): -179.146
log p(y | \bar{z}, eta): -179.119
log p(y | \bar{z}, eta): -179.094
log p(y | \bar{z}, eta): -179.072
log p(y | \bar{z}, eta): -179.052
log p(y | \bar{z}, eta): -179.035
E-M Iteration 2
100
log p(y | \bar{z}, eta): -164.228
log p(y | \bar{z}, eta): -56.7449
....
```

At this point, we load the trained lda model into a python session to inspect the
model parameters, namely $\alpha$, $\beta$, $\eta$.

```python
In [1]: import numpy as np

# The trained model is saved in numpy format
In [2]: with open("/tmp/fslda_model.npy") as model:
   ...:     alpha = np.load(model)
   ...:     beta = np.load(model)
   ...:     eta = np.load(model)

# Print the contents of alpha. We have trained our model with 10 topics and
# subsequently the shape of alpha is (10, 1)
In [3]: alpha
Out[3]:
array([[ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1],
       [ 0.1]])

# Print the contents of beta. The shape of beta is (10, 1000), as it refers to
# the per topic word distributions
In [4]: beta
Out[4]:
array([[ 0.00086267,  0.00094678,  0.00102146, ...,  0.00096579,
         0.00114778,  0.00068308],
       [ 0.00093683,  0.00057377,  0.00091816, ...,  0.00084292,
         0.0019941 ,  0.00158896],
       [ 0.00061499,  0.00052463,  0.0014251 , ...,  0.00058796,
         0.00103788,  0.00199141],
       ..., 
       [ 0.00127186,  0.00103715,  0.0012203 , ...,  0.00093232,
         0.0010174 ,  0.00059228],
       [ 0.00068713,  0.00077923,  0.0011483 , ...,  0.00078904,
         0.00100819,  0.00151192],
       [ 0.00110573,  0.00129864,  0.00132421, ...,  0.00071085,
         0.00068612,  0.00109131]])

# Print the contents of eta. The shape of eta is (10, 6)
In [5]: eta
Out[5]:
array([[ -1.36514771e+00,  -1.36315903e+00,   6.86343073e+00,
         -1.36160849e+00,  -1.40997983e+00,  -1.36353567e+00],
       [ -3.86338240e-02,  -4.22127522e-02,  -1.59781985e-02,
         -4.32016108e-02,   2.18725518e-01,  -7.86991329e-02],
       [ -1.43063101e-01,  -1.29929507e-01,  -1.56495059e-01,
         -1.41158423e-01,   7.40731106e-01,  -1.70085016e-01],
       [ -1.37109145e+00,   6.89966969e+00,  -1.37034253e+00,
         -1.36977176e+00,  -1.41835979e+00,  -1.37010416e+00],
       [  1.34353856e-02,   2.21828346e-02,  -3.55681181e-02,
         -1.37398303e-03,   5.41772921e-02,  -5.28534111e-02],
       [  6.89574101e+00,  -1.36937035e+00,  -1.36901212e+00,
         -1.36863390e+00,  -1.41848080e+00,  -1.37024383e+00],
       [ -1.35957482e+00,  -1.35852610e+00,  -1.35898905e+00,
         -1.35614477e+00,  -1.40582804e+00,   6.83906279e+00],
       [ -1.36908306e+00,  -1.36697518e+00,  -1.36868835e+00,
          6.89124942e+00,  -1.41688873e+00,  -1.36961410e+00],
       [ -1.39205753e+00,  -1.39198917e+00,  -1.39207285e+00,
         -1.39018710e+00,   6.95507809e+00,  -1.38877145e+00],
       [  3.02661927e-02,  -1.10847543e-02,  -4.02073659e-02,
         -1.21513799e-02,   4.85163748e-02,  -1.53390674e-02]])
```

Now, we use the trained model to transform our data with the following bash
command.

```bash
$ fslda transform /tmp/fslda_model.npy /tmp/data.npy /tmp/transformed_data.npy
E-M Iteration 1
100
```

If we load the transformed data into a python session, we can inspect the
transformed data. 

```python
In [1]: import numpy as np

In [2]: with open("/tmp/transformed_data.npy", "rb") as f:
   ...:     Z = np.load(f)

# Print the contents of Z, which is the per topic document distribution. We
# could say that Z[0] is the number of words that were produced from topic 0.
# The size of Z is (10, 100)
In [3]: Z
Out[3]:
array([[ 1019.9107783 ,   859.3746101 ,  1427.69863967,  1088.73008294,
         1071.07543482,   931.59113907,   960.15907305,   957.75190977,
          965.91898608,   836.01357053,   922.43651427,  1382.846077  ,
         1499.20517748,   985.486335  ,   855.54095167,   923.05499726,
         1082.24408841,  1570.02741798,   868.04180258,   909.57401057,
          985.84969404,   965.83696595,   922.3555769 ,  1031.53327711,
          872.40110706,  1012.21692746,  1001.2492608 ,   975.12520763,
         1064.25326642,  1088.55563596,   852.16611375,   945.48605892,
          932.40444614,   887.04821804,   857.11401409,   920.91309446,
          909.32228312,   981.6026501 ,  1013.32953975,  1038.88487482,
         1413.93500051,   925.30685048,  1055.40425636,   927.10705897,
         1011.45482269,   844.79497123,   942.21571161,   935.89654987,
          943.62254523,   914.57544146,   912.46043094,   896.38166963,
         1448.73039247,  1009.70394328,  1525.21304882,   909.14173508,
         1062.95047749,   934.87948091,   951.19660927,   942.72889388,
          879.37797766,  1417.91414612,  1132.91543207,   878.91869879,
         1009.12487562,   949.10501698,  1393.57410989,   902.35129597,
          905.41060513,   903.64511028,   913.05203735,   976.49796291,
          909.17071429,   937.19231767,  1014.7799845 ,   935.8190576 ,
          839.70289601,   917.40720139,   893.60382168,   976.66085359,
         1008.81373381,   943.51792376,   979.14676223,  1349.04931106,
         1497.64595248,   910.68349036,  1002.55971879,  1343.15446418,
          932.3673627 ,   907.67798956,   947.60219197,  1009.50945217,
         1541.55112037,  1063.06633006,   770.33119431,  1002.83059115,
         842.20809669,   827.84779943,   929.93643816,   936.53056171],
         ...
       [  902.27101254,   984.28387808,  1007.78414228,  1100.91715608,
          949.3457713 ,  1017.66400711,  1054.94456398,   948.59827311,
          962.48949314,  1152.11542039,  1067.85499488,   845.31201371,
          870.28596559,   841.30403717,  1012.17751733,   770.95283444,
          887.37539866,   808.36110518,   911.25710004,   982.74918076,
         1067.33584398,   827.27832198,  1258.46460689,  1053.07873508,
          942.12301473,  1012.54885641,   937.90863862,   857.55367228,
          928.18083491,   922.68155761,   912.44795628,   856.66729263,
         1121.06348606,   923.82482903,  1043.97398546,  1253.07989607,
         1009.62109852,  1058.3169289 ,  1063.13110928,   860.01654337,
          934.12084623,   792.02495381,   936.27498759,  1514.8669249 ,
         1046.74444506,   985.71877773,  1098.6587326 ,   921.00987809,
         1019.04304075,   843.62692052,   911.85102535,   829.22548004,
          940.56079349,  1242.39111332,  1006.41522999,  1012.84659058,
          932.97561594,   945.0677803 ,   916.24276567,  1178.88230077,
         1023.66414521,   915.45476272,   898.63585347,   813.64985559,
          975.51344143,  1296.96414965,  1030.2561018 ,   894.91694467,
          904.08443257,   821.61868474,  1140.823663  ,  1245.67698937,
          931.83417298,   928.39920503,  1052.85132391,   938.1737776 ,
         1058.24803129,  1009.62787825,   953.73255392,   968.49120585,
         1221.96227303,  1148.22065894,   979.75188157,   984.58752694,
         1147.54343614,   995.13025838,   925.14761445,   842.16401113,
          944.04983208,  1349.38277865,   860.47146176,  1079.81075318,
          799.40590558,  1098.63316048,  1039.41374872,  1003.07705429,
         1121.47772463,   773.91529544,  1152.11024316,  1168.75355131]])
```

lda application
===============

**lda** is the command-line program that implements the vanilla LDA
(unsupervised LDA), as it was introduced in [*Latent Dirichlet
Allocation*](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), by Blei
et al. This program allows the training of an unsupervised LDA model and the
application of that model to a set of data in order to transform them.

Apart from the optional arguments, which we have already discussed, the user
can specify additional arguments. The extensive help menu of **lda** is
presented below.

```bash
$ lda --help
Console application for unsupervised LDA.

    Usage:
        lda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                  [--e_step_tolerance=ET] [--random_state=RS]
                  [--compute_likelihood=CL] [--initialize_seeded | --initialize_random]
                  [-q | --quiet] [--snapshot_every=N] [--workers=W]
                  [--continue=M] DATA MODEL
        lda transform [-q | --quiet] [--e_step_iterations=EI]
                      [--e_step_tolerance=ET] [--workers=W]
                      MODEL DATA OUTPUT
        lda (-h | --help)

    General Options:
        -h, --help              Show this help
        -q, --quiet             Produce no output to the terminal
        --topics=K              How many topics to train [default: 100]
        --iterations=I          Run LDA for I iterations [default: 20]
        --random_state=RS       The initial seed value for any random numbers
                                needed [default: 0]
        --initialize_seeded     Initialize the topic over words distributions by
                                seeding them from the passed documents. The
                                default initialization option is initialize_seeded
        --initialize_random     Initialize the topic over words distributions as
                                random distributions. The default
                                initialization option is initialize_seeded
        --snapshot_every=N      Snapshot the model every N iterations [default: -1]
        --workers=N             The number of concurrent workers [default: 1]
        --continue=M            A model to continue training from

    E Step Options:
        --e_step_iterations=EI  The maximum number of iterations to perform
                                in the E step [default: 30]
        --e_step_tolerance=ET   The minimum accepted relative increase in log
                                likelihood during the E step [default: 1e-3]
        --compute_likelihood=CL The percentage of documents to compute the
                                likelihood for (1.0 means compute for every
                                document) [default: 0.0]
```

The user can specify the values of the following arguments:

- **e_step_iterations**: This argument sets the maximum number of times to
  alternate between maximizing $\gamma$ and $\phi$ in the Expectation step.

- **e_step_tolerance**: The minimum relative change between consecutive updates
  in the variational parameter $\gamma$. During the Expectation step, we try to
  learn the variational parameters $\phi$ and $\gamma$, by iteratively updating
  their values. However, even though we define a maximum number of iterations
  (via **e_step_iterations** argument), this process can stop earlier if the
  difference between the $\gamma$ in the $i^{th}$ iteration and the
  $\gamma$ in the $(i+1)^{th}$ iteration is less than
  **e_step_tolerance** (default=1e-3).

- **compute_likelihood**: The variational parameters of the trained model are
  updated according to every document in the corpus, during the
  Expectation-step. As soon as, the Expectation-step is complete we compute the
  value of the Evidence Lower Bound (ELBO). This value can be computed either
  for all the documents in the corpus or for a percentage of them. The
  percentage of documents to be used during this computation is given via the
  **compute_likelihood** argument. Obviously, 1.0 means compute for every
  document in the corpus (default=0.0).

slda application
================

**slda** is the command-line program that implements the supervised LDA (sLDA).
For more information about the sLDA, see papers [*"Simultaneous Image
Classification and
Annotation"*](http://www.cs.cmu.edu/~chongw/papers/WangBleiFeiFei2009.pdf), by
Wang et al. and [*"Supervised topic
models"*](https://www.cs.princeton.edu/~blei/papers/BleiMcAuliffe2007.pdf), by
Blei et al. This program allows the training of a supervised LDA model and the
application of that model to a set of data in order to transform them.

The extensive help menu of **slda** is presented below.

```bash
$ slda --help
Console application for supervised LDA.
    
    Usage:
        slda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                   [--e_step_tolerance=ET] [--fixed_point_iterations=FI]
                   [--random_state=RS] [--compute_likelihood=CL]
                   [--initialize_seeded | --initialize_random]
                   [--m_step_iterations=MI] [--m_step_tolerance=MT]
                   [--regularization_penalty=L]
                   [-q | --quiet] [--snapshot_every=N] [--workers=W]
                   [--continue=M] [--continue_from_unsupervised=M] DATA MODEL
        slda transform [-q | --quiet] [--e_step_iterations=EI]
                       [--e_step_tolerance=ET] [--workers=W]
                       MODEL DATA OUTPUT
        slda (-h | --help)

    General Options:
        -h, --help                        Show this help
        -q, --quiet                       Produce no output to the terminal
        --topics=K                        How many topics to train [default: 100]
        --iterations=I                    Run LDA for I iterations [default: 20]
        --random_state=RS                 The initial seed value for any random numbers
                                          needed [default: 0]
        --initialize_seeded               Initialize the topic over words distributions by
                                          seeding them from the passed documents. The
                                          default initialization option is initialize_seeded
        --initialize_random               Initialize the topic over words distributions as
                                          random distributions. The default
                                          initialization option is initialize_seeded
        --snapshot_every=N                Snapshot the model every N iterations [default: -1]
        --workers=N                       The number of concurrent workers [default: 1]
        --continue=M                      A model to continue training from
        --continue_from_unsupervised=M    An unsupervised model to continue training from

    E Step Options:
        --e_step_iterations=EI            The maximum number of iterations to perform
                                          in the E step [default: 10]
        --e_step_tolerance=ET             The minimum accepted relative increase in log
                                          likelihood during the E step [default: 1e-4]
        --fixed_point_iterations=FI       The number of fixed point iterations to compute
                                          phi [default: 20]
        --compute_likelihood=CL           The percentage of documents to compute the
                                          likelihood for (1.0 means compute for every
                                          document) [default: 0.0]

    M Step Options:
        --m_step_iterations=MI            The maximum number of iterations to perform
                                          in the M step [default: 200]
        --m_step_tolerance=MT             The minimum accepted relative increase in log
                                          likelihood during the M step [default: 1e-4]
        -L L, --regularization_penalty=L  The regularization penalty for the Multinomial
                                          Logistic Regression [default: 0.05]

```

In case of **slda** the user can continue the training of an unsupervised model
in a supervised manner, by using the **continue_from_unsupervised** argument to
define the path to the model to continue from.

- **fixed_point_iterations**: In supervised LDA, the update of $\phi$ is a
  fixed-point iteration method. The maximum number of iterations used in the
  maximization of $\phi$ can be specified via **fixed_point_iterations**
  argument.

- **m_step_iterations**: The maximum number of gradient descent iterations
  (default=200).

- **m_step_tolerance**: The minimum relative improvement between consecutive
  gradient descent iterations (default=1e-4).

- **regularization_penalty**: The L2 penalty for logistic regression
  (default=0.05).

The rest of the arguments, namely **e_step_iterations**, **e_step_tolerance**
and **compute_likelihood** are already explained in [lda](#lda-application).

fslda application
================

**fslda** is the command-line program that implements the fast supervised LDA
(fsLDA). For more information, please check our publication with title [*"Fast
Supervised LDA for Discovering Micro-Events in Large-Scale Video
Datasets"*](http://mug.ee.auth.gr/wp-content/uploads/fsLDA.pdf).

The extensive help menu of **fslda** is presented below.

```bash
$ fslda --help
Console application for fast supervised LDA (fsLDA).

    Usage:
        fslda train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                    [--e_step_tolerance=ET] [--random_state=RS]
                    [--compute_likelihood=CL] [--initialize_seeded | --initialize_random]
                    [--supervised_weight=C] [--m_step_iterations=MI]
                    [--m_step_tolerance=MT] [--regularization_penalty=L]
                    [-q | --quiet] [--snapshot_every=N] [--workers=W]
                    [--continue=M] [--continue_from_unsupervised=M] DATA MODEL
        fslda online_train [--topics=K] [--iterations=I] [--e_step_iterations=EI]
                           [--e_step_tolerance=ET] [--random_state=RS]
                           [--compute_likelihood=CL] [--initialize_seeded | --initialize_random]
                           [--supervised_weight=C] [--regularization_penalty=L] [--batch_size=BS]
                           [--momentum=MM] [--learning_rate=LR] [--beta_weight=BW]
                           [-q | --quiet] [--snapshot_every=N] [--workers=W]
                           [--continue=M] [--continue_from_unsupervised=M] DATA MODEL
        fslda transform [-q | --quiet] [--e_step_iterations=EI]
                        [--e_step_tolerance=ET] [--workers=W]
                        MODEL DATA OUTPUT
        fslda (-h | --help)

    General Options:
        -h, --help                        Show this help
        -q, --quiet                       Produce no output to the terminal
        --topics=K                        How many topics to train [default: 100]
        --iterations=I                    Run LDA for I iterations [default: 20]
        --random_state=RS                 The initial seed value for any random numbers
                                          needed [default: 0]
        --initialize_seeded               Initialize the topic over words distributions by
                                          seeding them from the passed documents. The
                                          default initialization option is initialize_seeded
        --initialize_random               Initialize the topic over words distributions as
                                          random distributions. The default
                                          initialization option is initialize_seeded
        --snapshot_every=N                Snapshot the model every N iterations [default: -1]
        --workers=N                       The number of concurrent workers [default: 1]
        --continue=M                      A model to continue training from
        --continue_from_unsupervised=M    An unsupervised model to continue training from

    E Step Options:
        --e_step_iterations=EI            The maximum number of iterations to perform
                                          in the E step [default: 10]
        --e_step_tolerance=ET             The minimum accepted relative increase in log
                                          likelihood during the E step [default: 1e-4]
        -C C, --supervised_weight=C       The weight of the supervised term for the
                                          E step [default: 1]
        --compute_likelihood=CL           The percentage of documents to compute the
                                          likelihood printed (1.0 means compute for every
                                          document) [default: 0.0]
    M Step Options:
        -L L, --regularization_penalty=L  The regularization penalty for the Multinomial
                                          Logistic Regression in M step [default: 0.05]

    Batch M Step Options:
        --m_step_iterations=MI            The maximum number of iterations to perform
                                          in the M step [default: 200]
        --m_step_tolerance=MT             The minimum accepted relative increase in log
                                          likelihood during the M step [default: 1e-4]

    Online M Step Options:
        --batch_size=BS                   The mini-batch size for the online learning [default: 128]
        --momentum=MM                     Set the momentum for changing eta [default: 0.9]
        --learning_rate=LR                Set the learning rate for changing eta [default: 0.01]
        --beta_weight=BW                  Set the weight of the previous beta parameters
                                          w.r.t to the new from the minibatch [default: 0.9]

```

In the case of **fsLDA** we added an additional command, named
**online_train**. This command chooses an *online learning method*, namely
instead of using all training data in each Expectation-Maximization update, we
use a mini-batch of the training data to update the parameters incrementally,
using *Stochastic Variational Bayes* for $\beta$ and *Stochastic Gradient
Descent* for $\eta$.

- **C**: One of the main contributions of **Fast Supervised LDA (fsLDA)**
  concerns the fact that this variation of LDA is able to adapt the influence
  of the supervised part on the final topic representation. This influence is
  quantified via an extra hyperparameter, named **C**, which is used for the
  weighting of the supervised component in the update rule of $\phi$
  (default=1).

- **batch_size**: The number of documents, after which we perform the
  Maximization step (default=128), in case of the online training method. 

- **momentum**: The momentum used for the SGD update of $\eta$, in case of the
  online training method (default=0.9).

- **learning_rate**: The learning rate used for the SGD update of $\eta$, in
  case of the online training method (default=0.01).

- **beta_weight**: The weight used for the online update of $\beta$ according
  to the following equation
  $\beta_{new} = \text{beta_weight}\*\beta_{old} + (1-\text{beta_weight})\*\beta_{batch}$
  (default=0.9).

The rest of the arguments, namely **e_step_iterations**, **e_step_tolerance**,
**compute_likelihood**, **regularization_penalty**, **m_step_iterations** and
**m_step_tolerance** are already explained in [lda](#lda-application) and
[slda](#slda-application).

Bash completion
---------------

In the [installation](/installation/#bash-completion) page you can also see how
you could install a provided bash completion script to autocomplete the options
and commands for faster and easier usage of the console applications.
