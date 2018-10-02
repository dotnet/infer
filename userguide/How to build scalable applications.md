---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## How to build scalable applications

Applications can be made scalable along three different axes: the number of data points, the number of discrete values taken by variables, and the number of elements in vectors. The following describes how to achieve scaling along each of these axes.

### Scaling the number of data points

If you want to apply the same model to many data points, and the number of data points exceeds the available memory, you have two options. The first option is to split up the data, reading one chunk into memory at a time, and cycling through the chunks until convergence. This approach is described on the page about [sharing variables between models](Sharing variables between models.md), and nicely illustrated in the examples of [Shared variable LDA](Shared variable LDA.md) and [Click model 2](Click model 2.md). This approach has the benefit that the answers will be same as if you had processed all data at once (modulo initialisation and convergence issues). The downside it that it requires repeated loading and processing of the data chunks, which can be slow.

The second approach aims to speed up inference by streaming through the data only once. This is called "online learning." The idea is to use the posterior distribution from each data chunk as the prior for the next chunk. To do this, you set up your model so that the prior distributions on the parameters can be easily changed at runtime (i.e. using observed values for the priors). The priors start off as the real priors, but after processing each data chunk, you change the priors to be the posterior from that chunk. The benefit of this approach is that it is very fast, requiring a single pass through the data. The downside is that the answers will not be the same as (generally they will be inferior to) processing the data together.

### Scaling the number of discrete values a variable can take

If your model contains discrete variables with a large number of values, then you can easily exceed memory limits when storing distributions on those values. In many cases, only a few values have significant probability, i.e. the distributions are sparse. You can instruct Infer.NET to exploit this sparsity by changing its representation of these distributions. This is described on the page about [using sparse messages](using sparse messages.md).

### Scaling the size of vectors

If your model contains random vectors with multivariate Gaussian distributions, then memory and time costs will increase significantly as the dimension of these vectors increases. This happens because the multivariate Gaussian stores a full covariance matrix over the vector. To make the inference scalable, you can instruct Infer.NET to approximate this covariance matrix with a diagonal matrix, i.e. to approximate the vector elements as independent in the posterior. This is done by using an array of doubles instead of a vector. Infer.NET automatically uses independent posteriors for elements of an array. You will need to convert each vector operation into the equivalent array operation, sometimes by expanding a single operation into multiple operations (for example, inner product must be converted into array-wise multiplication followed by a sum). You can mix and match arrays and vectors in the same model by using the [array-vector conversion operations](Miscellaneous factors.md).