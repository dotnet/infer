---
layout: default
---
 
[Infer.NET user guide :](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Variable types and their distributions

The following table shows what types of variables are supported by Infer.NET, along with the distributions which are available for representing uncertainty in each type. You can create variables for each of these types using [the static methods on Variable for each distribution](Distribution factors.md). Some distributions with experimental quality band are not shown in this table.

| **Variable type** | **Restrictions** | **Distribution** | **Distribution Class** | **Example of use** |
|---------------------------------------------------------------------------------------------|
| bool | - | [Bernoulli](http://en.wikipedia.org/wiki/Bernoulli_distribution) | [Bernoulli](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Bernoulli.html) | [Two coins tutorial](Two coins tutorial.md) |
| double | - | [Gaussian](http://en.wikipedia.org/wiki/Normal_distribution) | [Gaussian](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Gaussian.html) | [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md) |
| double | between 0 and infinity | [Gamma](http://en.wikipedia.org/wiki/Gamma_distribution) | [Gamma](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Gamma.html) | [Learning a Gaussian tutorial](Learning a Gaussian tutorial.md) |
| double | between 0 and 1 | [Beta](http://en.wikipedia.org/wiki/Beta_distribution) | [Beta](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Beta.html) | [Clinical trial tutorial](Clinical trial tutorial.md) |
| double | between settable lower and upper bounds | [Truncated Gaussian](http://en.wikipedia.org/wiki/Truncated_normal_distribution) | [TruncatedGaussian](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.TruncatedGaussian.html) | - |
| double | between 0 and settable period length | [Wrapped Gaussian](http://en.wikipedia.org/wiki/Wrapped_gaussian) | [WrappedGaussian](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.WrappedGaussian.html) | - |
| double | between a lower bound and infinity | [Pareto](http://en.wikipedia.org/wiki/Pareto_distribution) | [Pareto](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Pareto.html) | - |
| int | between 0 and D-1 inclusive | [Discrete](http://en.wikipedia.org/wiki/Categorical_distribution) (categorical) | [Discrete](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Discrete.html) | [Latent Dirichlet Allocation](Latent Dirichlet Allocation.md) |
| int | between 0 and infinity | [Poisson](http://en.wikipedia.org/wiki/Poisson_distribution) | [Poisson](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Poisson.html) | - |
| int | between 0 and N inclusive | [Binomial](http://en.wikipedia.org/wiki/Binomial_distribution) | [Binomial](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Binomial.html) | - |
| enum | - | Discrete over enum values | [DiscreteEnum](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DiscreteEnum-1.html) | - |
| Vector | - | [Vector Gaussian](http://en.wikipedia.org/wiki/Multivariate_normal_distribution) | [VectorGaussian](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.VectorGaussian.html) | [Mixture of Gaussians tutorial](Mixture of Gaussians tutorial.md) |
| Vector | each element between 0 and 1, elements sum to 1 | [Dirichlet](http://en.wikipedia.org/wiki/Dirichlet_distribution) | [Dirichlet](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Dirichlet.html) | [Latent Dirichlet Allocation](Latent Dirichlet Allocation.md) |
| PositiveDefiniteMatrix | matrix is positive definite | [Wishart](http://en.wikipedia.org/wiki/Wishart_distribution) | [Wishart](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Wishart.html) | [Mixture of Gaussians tutorial](Mixture of Gaussians tutorial.md) |
| string | - | [Probabilistic automaton](http://en.wikipedia.org/wiki/Probabilistic_automaton) | [StringDistribution](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.StringDistribution.html) | [Hello, Strings!](Hello, Strings!.md) |
| char | - | Discrete over char values | [DiscreteChar](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DiscreteChar.html) | - |
| TDomain[] |  T is a value-type  distribution over a domain TDomain | Array of distributions considered as a distribution over an array | [DistributionStructArray<T, TDomain>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DistributionStructArray-2.html) | - |
| TDomain[,] | T is a value-type distribution over a domain TDomain | 2-D Array of distributions considered as a distribution over a 2-D array | [DistributionStructArray2D<T, TDomain>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DistributionStructArray2D-2.html) | - |
| TDomain[] | T is a reference-type distribution over a domain TDomain | Array of distributions considered as a distribution over an array | [DistributionRefArray<T, TDomain>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DistributionRefArray-2.html) | - |
| TDomain[,] | T is a reference-type distribution over a domain TDomain | 2-D Array of distributions considered as a distribution over a 2-D array | [DistributionRefArray2D<T, TDomain>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DistributionRefArray2D-2.html) | - |
| ISparseList<bool\> | - | Sparse list of [Bernoulli](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Bernoulli.html) distributions considered as a distribution over a sparse list of bools | [SparseBernoulliList](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SparseBernoulliList.html) | - |
| ISparseList<double\> | elements between 0 and 1 | Sparse list of [Beta](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Beta.html) distributions considered as a distribution over a sparse list of doubles | [SparseBetaList](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SparseBetaList.html) | - |
| ISparseList<double\> | - | Sparse list of [Gaussian](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Gaussian.html) distributions considered as a distribution over a sparse list of doubles | [SparseGaussianList](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SparseGaussianList.html) | - |
| ISparseList<double\> | elements between 0 and infinity | Sparse list of [Gamma](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.Gamma.html) distributions considered as a distribution over a sparse list of doubles | [SparseGammaList](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SparseGammaList.html) | - |
| IList<int\> | elements between 0 and N-1 inclusive | SparseBernoulliList where domain is list of indices with value true | [BernoulliIntegerSubset](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.BernoulliIntegerSubset.html) | - |
| IFunction | - | [Sparse Gaussian Process](http://en.wikipedia.org/wiki/Gaussian_process) | [SparseGP](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.SparseGP.html) | [Gaussian process classifier](Gaussian Process classifier.md) |

**_Notes:_**

*   For descriptions of the `Vector` and `PositiveDefiniteMatrix` see the page on [Vector and Matrix types](Vector and matrix types.md).
*   [DistributionRefArray<T, TDomain>](../apiguide/api/Microsoft.ML.Probabilistic.Distributions.DistributionRefArray-2.html) can be used to represent a distribution over an arbitrarily deep jagged array domain. For example, the following alias (which can be copied and pasted into you code) represents a 2-deep array of Gaussians considered as a distribution over a 2-deep jagged array of double:

    ```csharp
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    ```

*   Posterior distributions for array variables can be passed back as either .NET arrays of distributions (for example `Gaussian[][]`), or as distribution arrays (for example `GaussianArrayArray` using the above alias). The former can be achieved by using (for example) `Gaussian[][]` as the type parameter in the Infer method. The latter is the native format which is therefore is more efficient and needs no casting or type parameter. 
*   `IFunction` is an interface type which is used as the domain type for a `SparseGP` distribution. This interface has a single Evaluate method for a `Vector` domain:

    ```csharp
    double Evaluate(Vector v);
    ```
