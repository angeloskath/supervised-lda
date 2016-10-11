Visualize topics from MNIST dataset
===================================

The following example aims to point out the differences between the inferred
topics of LDA and fsLDA. We decided to use the [MNIST
database](http://yann.lecun.com/exdb/mnist/) which is a dataset of 70000
handwritten digits, in order to make the topics visualization more fancy!!! Both
models are trained using the console applications that are thoroughly explained
in the corresponding [documentation page](/console-applications/).

The console applications that are used in this example expect and save data in
**numpy format**. We have already transformed the MNIST dataset to numpy format
and you can download it if you copy and paste the following instructions in a
terminal.

```bash
$ cd /tmp
$ wget "http://ldaplusplus.com/files/mnist.tar.gz"
$ tar -zxf mnist.tar.gz
```

Inspect MNIST dataset
---------------------

After untaring the `mnist.tar.gz`, the extracted files are the
`mnist_train.npy` and `mnist_test.npy`, which correspond to the training and
test data respectively. At this point, we load the data into a python session
to inspect them.

```python
In [1]: import numpy as np

# We load the training set to inspect them
In [2]: with open("mnist_train.npy", "rb") as f:
   ...:     X_train = np.load(f)
   ...:     y_train = np.load(f)

# Print the shapes of X_train and y_train 
In [3]: X_train.shape
Out[3]: (784, 60000)

In [4]: y_train.shape
Out[4]: (60000,)
```

We observe that the training set is an array of size (784, 60000). All images
in the dataset are $28\times28$ images, thus the first dimension is 784
(28*28=784), while the second dimension refers to the number of the training
samples, which is 60000.

In the subsequent python session, we use the [*matplotlib
library*](http://matplotlib.org/) and the [*seaborn visualization
library*](https://stanford.edu/~mwaskom/software/seaborn/) to plot 20 randomly
selected training image.

```python
In [5]: import matplotlib.pyplot as plt

In [6]: import seaborn as sns

# Set the aesthetic style of the plots
In [7]: sns.set_style("dark")

# Create 20 subplots
In [8]: fig, axes = plt.subplots(2, 10, figsize=(10, 2))

In [9]: for i in xrange(2):
   ...:     for j in xrange(10):
   ...:         axes[i][j].imshow(X_train[:, np.random.randint(0, 6000)].reshape(28, 28), cmap='gray_r', interpolation='nearest')
   ...:         axes[i][j].set_xticks([])
   ...:         axes[i][j].set_yticks([])
```

One possible output could be the following image.

<figure>
    <img src="/img/mnist-example/mnist_training_samples.svg"
         alt="MNIST training samples"
         class="full-width" />
    <figcaption>MNIST sample images from training set</figcaption>
</figure>

Both in LDA and fsLDA, each document can be viewed as a mixture of various
topics. However, the MNIST database consists exclusively of images, thus there
is no direct analogy between words and documents. In order to use the MNIST
dataset in combination with LDA and fsLDA we consider each image to be a
document and each pixel in the image to be a word. As a result, our corpus
consists of 70000 documents and the vocabulary size is 784.

Unsupervised LDA
----------------

In this section, we will visualize the inferred topics from an unsupervised LDA
model using the [*lda*](/console-applications/#lda-application) application. We train an
unsupervised model with 10 topics for 20 Expectation-Maximization steps. In
addition, we use the `snapshot_every` option to save a model after every 2
iterations so that we can inspect the topics evolution during the training
process. Furthermore, we select, randomly, the pseudo-random number generator
seed control to be 42.

```bash
$ lda train mnist_train.npy mnist_lda_model.npy --topics 10 --iterations 50 --random_state 42 --workers 4 --e_step_iterations 50 --snapshot_every 5
E-M Iteration 1
100
200
300
400
...
60000
E-M Iteration 2
100
200
300
...
59700
59800
59900
60000
E-M Iteration 15
100
200
...
60000
```

At this point, it is important to recall that the lda application trains a
model that consists of two arrays. The first one corresponds to $\alpha$, which
is the Dirichlet prior on the per-document topic distributions, while the
second one is $\beta$, which is the per-topic word distribution. Thus we can
visualize the topics, by merely visualizing the $\beta$ parameter. In the
following python session, we visualize the evolution of the inferred topics,
during the training process.

```python
In [1]: import numpy as np

In [2]: import matplotlib.pyplot as plt


```


Fast Supervised LDA
-------------------

Comparisons
-----------
