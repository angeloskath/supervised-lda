MNIST Example
=============

In the current example we use the [MNIST
database](http://yann.lecun.com/exdb/mnist/) which is a dataset of 70,0000
handwritten digits to train and inspect three trained LDA models, one for each
LDA variant. The training of the LDA modes is done using the console
applications that are thoroughly explained in the corresponding
[documentation page](/console-applications/).

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

We observe that the training data is an array of size (`784, 60000). All images
in the dataset are $28\times28$ images, thus the first dimension is 784
(28*28=784), while the second dimension refers to the number of the training
samples.

In the subsequent python session, we use the [**matplotlib
library**](http://matplotlib.org/) and the [**seaborn visualization
library**](https://stanford.edu/~mwaskom/software/seaborn/) to plot 20 randomly
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

Unsupervised LDA
----------------

In this section, we will train an unsupervised LDA model, using the [**lda
console application**](/console-applications/).

Supervised LDA
--------------

Fast Supervised LDA
-------------------

Comparisons
-----------
