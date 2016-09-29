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
# type of array.
In [2]: X = np.round(np.maximum(0, np.random.rand(1000, 100)*20)).astype(np.int32)

# Create the class labels for all documents in the corpus. We multiply with 5,
# because we assume that there are 6 classes.
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
$ ./fslda train /tmp/data.npy /tmp/fslda_model.npy --topics 10
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

# Print the contents of alpha. We have trained our model with 10 topics, as a
# result the shape of alpha is (10, 1)
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
# the per topic word distributions.
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

# Print the contents of eta. The shape of eta is (10, 6).
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
$ ./fslda transform /tmp/fslda_model.npy /tmp/data.npy /tmp/transform_data.npy
E-M Iteration 1
100
```

If we load the transformed data into a python session, we can inspect the
transformed data. 

```python
In [1]: import numpy as np

In [2]: with open("/tmp/transform_data.npy", "rb") as f:
   ...:     Z = np.load(f)

# Print the contents of Z, which is the per topic document distribution. We
# could say that Z[0] is the number of words that were produced from topic 0.
# The size of zeta is (10, 100).
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
